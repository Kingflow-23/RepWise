"""Client for Photon, a free geocoder built on OpenStreetMap data.

Photon needs no API key. Two endpoints are used:

- ``/api?q=...``: turn a place name ("Lyon Part-Dieu") into coordinates;
- ``/reverse?lat=..&lon=..&osm_tag=..&radius=..``: list map features with a
  given OpenStreetMap tag around a point (gyms, sports shops, ...).

Privacy: every call sends the place name or coordinates to the Photon service
configured by ``REPWISE_PHOTON_URL`` (``https://photon.komoot.io`` by default).
Map data lists places, not their stock or opening hours.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class PlacesUnavailableError(Exception):
    """The map service could not answer, so nothing is known about nearby places."""


class PhotonClient:
    """A small, synchronous Photon client with explicit errors."""

    def __init__(
        self,
        base_url: str,
        user_agent: str,
        timeout_seconds: float = 10,
        transport: httpx.BaseTransport | None = None,
    ):
        """Configure the service.

        Args:
            base_url: Photon base URL, without a trailing slash.
            user_agent: User-Agent header; public services ask for one.
            timeout_seconds: Timeout for each HTTP request.
            transport: Optional ``httpx`` transport, used by tests to answer
                without the network.
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(
            timeout=timeout_seconds,
            headers={"User-Agent": user_agent},
            transport=transport,
        )

    def _get(self, path: str, params: dict[str, object]) -> list[dict]:
        """Call one endpoint and return its GeoJSON features.

        Args:
            path: ``/api`` or ``/reverse``.
            params: Query parameters.

        Returns:
            The ``features`` list (possibly empty).

        Raises:
            PlacesUnavailableError: On a network error, an HTTP error status,
                or a response that is not GeoJSON.
        """
        try:
            response = self.client.get(f"{self.base_url}{path}", params=params)
            response.raise_for_status()
            features = response.json().get("features", [])
        except (httpx.HTTPError, ValueError) as error:
            logger.warning("Photon request failed path=%s error=%s", path, error)
            raise PlacesUnavailableError(
                "The map service did not answer, so no places could be checked."
            ) from error
        if not isinstance(features, list):
            raise PlacesUnavailableError("The map service sent an unexpected answer.")
        return features

    def geocode(self, query: str, limit: int = 10) -> list[dict]:
        """Find the places matching a name, best match first, worldwide.

        The search is deliberately not limited to an area: limited to France,
        Photon answers "Berlin" with the French village of Berling. Callers
        choose among the matches instead.

        Args:
            query: Address, city or landmark.
            limit: Maximum number of matches.

        Returns:
            GeoJSON features (possibly empty).

        Raises:
            PlacesUnavailableError: If the service cannot be used.
        """
        return self._get("/api", {"q": query, "limit": limit})

    def search_nearby(
        self,
        osm_tag: str,
        latitude: float,
        longitude: float,
        radius_km: int,
        limit: int,
    ) -> list[dict]:
        """List map features with one OpenStreetMap tag around a point.

        Args:
            osm_tag: Tag such as ``leisure:fitness_centre``.
            latitude: Centre latitude.
            longitude: Centre longitude.
            radius_km: Search radius in kilometres.
            limit: Maximum number of features to request.

        Returns:
            GeoJSON features, in the service's order (not sorted by distance).

        Raises:
            PlacesUnavailableError: If the service cannot be used.
        """
        return self._get(
            "/reverse",
            {
                "lat": latitude,
                "lon": longitude,
                "radius": radius_km,
                "limit": limit,
                "osm_tag": osm_tag,
            },
        )
