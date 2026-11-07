"""Location tools (Location Agent): where the student is, and what is nearby.

``get_location`` decides which location to use, in this order:

1. a place name the student gave in this request (geocoded with Photon);
2. a location the client shared for this request only (``session_location``);
3. the location the student chose to save (opt-in, ``saved_locations`` table).

If none is available, it says so: RepWise never guesses where someone is.

Every location must be coherent and inside the service area of
``config/places.json`` (metropolitan France for now): a place name resolves
to the map service's best match, which must be inside the area: "Berlin" is
refused rather than replaced by a French look-alike, a name as broad as
a country or a region is refused (a 5 km search around "France" would be
meaningless), and shared or saved coordinates outside the area are refused
instead of being searched.

``search_nearby_places`` searches one category from ``config/places.json``
around that location and returns only what the map service returned: names,
addresses, coordinates, a distance computed by code, and the data source.
Places outside the service area, or whose name shows they are unrelated to
strength training (a fencing shop, a yoga studio, ...), are dropped by code
before the model sees the list, and the list is sorted nearest first.
It never claims a shop has a product in stock, or that a place is open.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from typing import Literal


from config.config import PLACES_CONFIG_PATH
from src.integrations.photon import PlacesUnavailableError
from src.schemas.models import Model
from src.tools.context import Location, ToolContext
from src.tools.results import ToolFailure, failure, report_storage_errors

logger = logging.getLogger(__name__)

PlaceCategory = Literal["gym", "sports_store", "supplement_store"]
EARTH_RADIUS_KM = 6371


@lru_cache(maxsize=1)
def places_settings() -> dict:
    """Load ``config/places.json`` once.

    Returns:
        The decoded settings: categories and their OpenStreetMap tags, search
        limits and the attribution text.
    """
    return json.loads(PLACES_CONFIG_PATH.read_text(encoding="utf-8"))


def normalize_name(text: str) -> str:
    """Lower-case a name, drop accents and punctuation, for whole-word matching.

    Args:
        text: A place name or a hint, such as "Bio c' Bon" or "Équitation".

    Returns:
        Space-separated plain words, such as ``"bio c bon"``.
    """
    plain = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in plain if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", plain))


def name_matches(name: str, hints: list[str]) -> bool:
    """Say whether a name contains one of the hints as whole words.

    Whole words keep "run" from matching "Brunch Fitness".

    Args:
        name: Place name.
        hints: Words or phrases from ``config/places.json``.

    Returns:
        ``True`` when at least one hint appears.
    """
    words = f" {normalize_name(name)} "
    return any(f" {normalize_name(hint)} " in words for hint in hints if hint.strip())


def in_service_area(latitude: float, longitude: float) -> bool:
    """Say whether a point lies inside the configured service area.

    Args:
        latitude: Latitude in degrees.
        longitude: Longitude in degrees.

    Returns:
        ``True`` inside the area's bounding box.
    """
    min_lon, min_lat, max_lon, max_lat = places_settings()["service_area"]["bbox"]
    return min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon


def outside_area_failure(label: str) -> ToolFailure:
    """Build the failure for a location RepWise does not serve.

    Args:
        label: The location's name, for the message.

    Returns:
        An ``outside_service_area`` failure.
    """
    area = places_settings()["service_area"]["label"]
    return failure(
        "outside_service_area",
        f"{label!r} is outside {area}, the only area RepWise searches for now. "
        f"Ask the student for a town or an address in {area}.",
    )


def resolve_place(places: object, text: str) -> Location | ToolFailure:
    """Turn a place name into a coherent location inside the service area.

    Args:
        places: The map client (``PhotonClient`` or a test double), or ``None``.
        text: What the student typed: a town, a postcode with a town, an
            address or a landmark.

    Returns:
        The best match inside the area, or a failure: ``invalid_request``
        (empty or one-letter text), ``place_too_vague`` (a country or region),
        ``outside_service_area`` (the best match is elsewhere),
        ``place_not_found``, ``places_unavailable``.
    """
    settings = places_settings()
    area = settings["service_area"]
    query = " ".join(text.split())
    if len(normalize_name(query)) < 2:
        return failure(
            "invalid_request",
            "Give a town, a postcode with the town, or an address.",
        )
    if places is None:
        return failure("places_unavailable", "No map service is configured.")
    try:
        features = places.geocode(query)
    except PlacesUnavailableError as error:
        return failure("places_unavailable", str(error))
    best = next((f for f in features if f.get("geometry", {}).get("coordinates")), None)
    if best is None:
        return failure(
            "place_not_found",
            f"No place called {query!r} was found. Ask the student for a town, a "
            f"postcode with the town, or an address in {area['label']}.",
        )
    # Only the best match counts: picking a lower-ranked French match would
    # turn "Bruxelles" into a hamlet of that name. A postcode makes the
    # intended French place the best match.
    properties = best.get("properties", {})
    longitude, latitude = best["geometry"]["coordinates"]
    country = str(properties.get("countrycode", "")).upper()
    codes = {code.upper() for code in area["country_codes"]}
    if (country and country not in codes) or not in_service_area(latitude, longitude):
        refused = outside_area_failure(query)
        refused.error += " If a French place of that name was meant, add its postcode."
        return refused
    if properties.get("type") in settings["too_broad_place_types"]:
        return failure(
            "place_too_vague",
            f"{query!r} is a whole country or region, too broad for a nearby "
            "search. Ask the student for a town, a postcode with the town, "
            "or an address.",
        )
    label = ", ".join(
        str(properties[key])
        for key in ("name", "postcode", "city")
        if properties.get(key)
    )
    return Location(label=label or query, latitude=latitude, longitude=longitude)


class LocationResult(Model):
    """Output of ``get_location``."""

    ok: Literal[True] = True
    location: Location
    source: Literal["place_name", "session", "saved"]
    note: str


class Place(Model):
    """One place the map service returned."""

    name: str
    category: PlaceCategory
    distance_km: float
    latitude: float
    longitude: float
    address: str | None = None
    map_url: str | None = None
    name_suggests_fitness_equipment: bool = False


class PlacesResult(Model):
    """Output of ``search_nearby_places``, nearest place first."""

    ok: Literal[True] = True
    category: PlaceCategory
    near: Location
    radius_km: int
    places: list[Place]
    data_source: str


def distance_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Return the great-circle (haversine) distance between two points.

    Args:
        a_lat: Latitude of the first point.
        a_lon: Longitude of the first point.
        b_lat: Latitude of the second point.
        b_lon: Longitude of the second point.

    Returns:
        Distance in kilometres.
    """
    d_lat, d_lon = radians(b_lat - a_lat), radians(b_lon - a_lon)
    h = (
        sin(d_lat / 2) ** 2
        + cos(radians(a_lat)) * cos(radians(b_lat)) * sin(d_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * asin(sqrt(h))


def _address(properties: dict) -> str | None:
    """Join the address parts a GeoJSON feature has.

    Args:
        properties: The feature's ``properties``.

    Returns:
        "12 Rue X 75011 Paris", or ``None`` when no part is known.
    """
    parts = [
        properties.get(key) for key in ("housenumber", "street", "postcode", "city")
    ]
    return " ".join(str(part) for part in parts if part) or None


def _map_url(properties: dict) -> str | None:
    """Link to the OpenStreetMap object, so the student can check it.

    Args:
        properties: The feature's ``properties``.

    Returns:
        An openstreetmap.org URL, or ``None`` without an OSM ID.
    """
    kinds = {"N": "node", "W": "way", "R": "relation"}
    kind, identifier = kinds.get(properties.get("osm_type", "")), properties.get(
        "osm_id"
    )
    return (
        f"https://www.openstreetmap.org/{kind}/{identifier}"
        if kind and identifier
        else None
    )


@report_storage_errors
def get_location(
    context: ToolContext, place: str | None = None
) -> LocationResult | ToolFailure:
    """Resolve the location to search around, without storing anything.

    Args:
        context: Trusted tool context.
        place: A place name the student gave in this request, or ``None``.

    Returns:
        The location and where it came from, or a failure
        (``location_unknown``, ``place_not_found``, ``place_too_vague``,
        ``outside_service_area``, ``invalid_request``, ``places_unavailable``).
    """
    if place and place.strip():
        found = resolve_place(context.places, place)
        if isinstance(found, ToolFailure):
            return found
        return LocationResult(
            location=found,
            source="place_name",
            note="The place name was sent to the map service to find it.",
        )
    if context.session_location is not None:
        shared = context.session_location
        if not in_service_area(shared.latitude, shared.longitude):
            return outside_area_failure(shared.label)
        return LocationResult(
            location=context.session_location,
            source="session",
            note="Shared by the app for this request only; it is not stored.",
        )
    saved = context.repository.get_location(context.student_id)
    if saved is not None:
        if not in_service_area(saved["latitude"], saved["longitude"]):
            return outside_area_failure(saved["label"])
        return LocationResult(
            location=Location(
                label=saved["label"],
                latitude=saved["latitude"],
                longitude=saved["longitude"],
            ),
            source="saved",
            note="The location the student chose to save in RepWise.",
        )
    return failure(
        "location_unknown",
        "The student's location is unknown. Ask for a place name, or for permission "
        "to use their location.",
    )


def search_nearby_places(
    context: ToolContext,
    category: PlaceCategory,
    place: str | None = None,
    radius_km: int | None = None,
) -> PlacesResult | ToolFailure:
    """Find gyms, sports equipment shops or supplement shops near the student.

    Args:
        context: Trusted tool context.
        category: ``gym``, ``sports_store`` (equipment) or ``supplement_store``.
        place: Optional place name to search around instead of the student's
            own location (see :func:`get_location`).
        radius_km: Search radius; defaults and maximum come from
            ``config/places.json``.

    Returns:
        Places sorted nearest first, without unrelated or out-of-area ones,
        or a failure (any ``get_location`` failure, ``invalid_request``,
        ``places_unavailable``).
    """
    settings = places_settings()
    categories = settings["categories"]
    if category not in categories:
        return failure("invalid_request", f"Unknown category {category!r}.")
    radius = radius_km or settings["default_radius_km"]
    if not 1 <= radius <= settings["max_radius_km"]:
        return failure(
            "invalid_request",
            f"radius_km must be between 1 and {settings['max_radius_km']}.",
        )
    resolved = get_location(context, place)
    if isinstance(resolved, ToolFailure):
        return resolved
    if context.places is None:
        return failure("places_unavailable", "No map service is configured.")
    near = resolved.location

    try:
        features = [
            feature
            for tag in categories[category]["osm_tags"]
            for feature in context.places.search_nearby(
                tag, near.latitude, near.longitude, radius, limit=50
            )
        ]
    except PlacesUnavailableError as error:
        return failure("places_unavailable", str(error))

    hints = categories[category].get("fitness_name_hints", [])
    unrelated = categories[category].get("exclude_name_hints", [])
    found: dict[tuple[str, float, float], Place] = {}
    for feature in features:
        properties = feature.get("properties", {})
        name = properties.get("name")
        coordinates = feature.get("geometry", {}).get("coordinates")
        if not name or not coordinates:
            continue
        longitude, latitude = coordinates
        km = round(distance_km(near.latitude, near.longitude, latitude, longitude), 2)
        if (
            km > radius
            or not in_service_area(latitude, longitude)
            or name_matches(name, unrelated)
        ):
            continue
        # One place can carry two of the category's tags; keep it once.
        found[(name, round(latitude, 5), round(longitude, 5))] = Place(
            name=name,
            category=category,
            distance_km=km,
            latitude=latitude,
            longitude=longitude,
            address=_address(properties),
            map_url=_map_url(properties),
            name_suggests_fitness_equipment=name_matches(name, hints),
        )
    nearest = sorted(found.values(), key=lambda item: item.distance_km)
    return PlacesResult(
        category=category,
        near=near,
        radius_km=radius,
        places=nearest[: settings["max_results"]],
        data_source=settings["attribution"],
    )
