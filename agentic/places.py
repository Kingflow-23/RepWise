"""Find gyms and shops near a location, using OpenStreetMap data from Photon.

Map data lists shops, not their stock, so it cannot confirm a product is available.
"""

from __future__ import annotations

import logging
import re
from math import asin, cos, radians, sin, sqrt
from typing import Literal

import httpx
from pydantic import Field

from src.schemas.models import Model

logger = logging.getLogger(__name__)

PlaceCategory = Literal["gym", "sports_store", "supplement_store"]

PHOTON_URL = "https://photon.komoot.io/reverse"
USER_AGENT = "RepWise-course-project"
TIMEOUT_SECONDS = 10
MAX_RESULTS = 8
OSM_TAGS = {
    "gym": ["leisure:fitness_centre"],
    "sports_store": ["shop:sports"],
    "supplement_store": ["shop:nutrition_supplements", "shop:health_food"],
}
# The nearest sports shop can specialize in running or fencing, so for equipment we
# also name the nearest one that sounds like a fitness shop. This reads names only.
FITNESS_SHOP_NAMES = re.compile(
    r"fitness|muscu|d[ée]cathlon|intersport|go sport|sport 2000", re.IGNORECASE
)
NOTE = (
    "Data from OpenStreetMap contributors. It may be out of date, may include private "
    "venues, and does not show stock or whether a place is open, so check first."
)


class Location(Model):
    """Where the student is, supplied by the session and never by the model."""

    label: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class Place(Model):
    """One place found near the student."""

    name: str
    distance_km: float
    address: str | None = None


class PlacesResult(Model):
    """Output of ``find_nearby_places``, nearest place first."""

    ok: Literal[True] = True
    category: PlaceCategory
    near: str
    radius_km: int
    places: list[Place]
    try_first: str | None = None
    note: str = NOTE


class PlacesFailure(Model):
    """Output of ``find_nearby_places`` when it could not search."""

    ok: Literal[False] = False
    code: Literal["location_unknown", "places_unavailable"]
    error: str


def distance_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Return the great-circle distance between two points."""
    d_lat, d_lon = radians(b_lat - a_lat), radians(b_lon - a_lon)
    h = (
        sin(d_lat / 2) ** 2
        + cos(radians(a_lat)) * cos(radians(b_lat)) * sin(d_lon / 2) ** 2
    )
    return 2 * 6371 * asin(sqrt(h))


def _search_photon(tag: str, location: Location, radius_km: int) -> list[dict]:
    """Return the map features with one tag around the location."""
    response = httpx.get(
        PHOTON_URL,
        params={
            "lat": location.latitude,
            "lon": location.longitude,
            "radius": radius_km,
            "limit": 50,
            "osm_tag": tag,
        },
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("features", [])


def _to_place(feature: dict, location: Location) -> Place | None:
    """Turn a map feature into a place, or None if it has no name."""
    properties = feature["properties"]
    if not properties.get("name"):
        return None
    longitude, latitude = feature["geometry"]["coordinates"]
    parts = [
        properties.get(key) for key in ("housenumber", "street", "postcode", "city")
    ]
    return Place(
        name=properties["name"],
        distance_km=round(
            distance_km(location.latitude, location.longitude, latitude, longitude), 1
        ),
        address=" ".join(str(part) for part in parts if part) or None,
    )


def find_nearby_places(
    category: PlaceCategory, location: Location | None, radius_km: int = 5
) -> PlacesResult | PlacesFailure:
    """Search near ``location``. The location is sent to the map service."""
    if location is None:
        return PlacesFailure(
            code="location_unknown",
            error="The student's location is unknown. Ask them to share it.",
        )
    try:
        features = [
            feature
            for tag in OSM_TAGS[category]
            for feature in _search_photon(tag, location, radius_km)
        ]
    except (httpx.HTTPError, ValueError) as error:
        logger.warning("Map search failed category=%s error=%s", category, error)
        return PlacesFailure(
            code="places_unavailable",
            error="The map service did not answer, so no places could be checked.",
        )

    found = [_to_place(feature, location) for feature in features]
    nearest = sorted(
        (place for place in found if place and place.distance_km <= radius_km),
        key=lambda place: place.distance_km,
    )[:MAX_RESULTS]
    try_first = None
    if category == "sports_store":
        try_first = next(
            (place.name for place in nearest if FITNESS_SHOP_NAMES.search(place.name)),
            None,
        )
    return PlacesResult(
        category=category,
        near=location.label,
        radius_km=radius_km,
        places=nearest,
        try_first=try_first,
    )
