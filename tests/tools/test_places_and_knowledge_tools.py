"""Location tools against a mocked Photon service, and document search."""

import httpx
import pytest

from src.integrations.photon import PhotonClient
from src.retrieval.query_expansion import ScopeRefusalError
from src.schemas.models import RetrievedChunk
from src.tools import knowledge, places
from src.tools.context import Location
from tests.conftest import make_context

HOME = Location(label="Home", latitude=48.79, longitude=2.33)


def feature(name, latitude, longitude=2.33, **properties):
    return {
        "properties": {"name": name, "osm_type": "N", "osm_id": 42, **properties},
        "geometry": {"coordinates": [longitude, latitude]},
    }


def photon(handler):
    """A Photon client answered by ``handler`` instead of the network."""
    return PhotonClient(
        "https://photon.test", "tests", transport=httpx.MockTransport(handler)
    )


def context_with(demo, handler, **extra):
    return make_context(
        demo.repository, demo.student_id, places=photon(handler), **extra
    )


# --- get_location ------------------------------------------------------------


def test_a_named_place_is_geocoded(demo):
    def handler(request):
        assert request.url.path == "/api" and request.url.params["q"] == "Lyon"
        return httpx.Response(
            200, json={"features": [feature("Lyon", 45.76, 4.83, country="France")]}
        )

    result = places.get_location(context_with(demo, handler), "Lyon")

    assert result.source == "place_name"
    assert (result.location.latitude, result.location.longitude) == (45.76, 4.83)


def test_the_session_location_comes_before_the_saved_one(demo):
    demo.repository.save_location(demo.student_id, "Saved", 45.76, 4.83)
    context = context_with(
        demo, lambda request: httpx.Response(500), session_location=HOME
    )

    assert places.get_location(context).source == "session"
    assert (
        places.get_location(context_with(demo, lambda r: httpx.Response(500))).source
        == "saved"
    )


def test_an_unknown_location_is_never_guessed(demo):
    result = places.get_location(
        context_with(demo, lambda request: httpx.Response(500))
    )

    assert result.code == "location_unknown"


def test_an_unknown_place_name_is_reported(demo):
    result = places.get_location(
        context_with(demo, lambda r: httpx.Response(200, json={"features": []})),
        "Atlantis",
    )

    assert result.code == "place_not_found"


# --- search_nearby_places -------------------------------------------------------


def test_places_are_sorted_filtered_deduplicated_and_linked(demo):
    def handler(request):
        if request.url.params["osm_tag"] != "leisure:fitness_centre":
            return httpx.Response(200, json={"features": [feature("Near gym", 48.791)]})
        return httpx.Response(
            200,
            json={
                "features": [
                    feature("Far gym", 48.83),
                    feature("Near gym", 48.791),
                    feature("Too far", 49.5),
                    {"properties": {}, "geometry": {"coordinates": [2.33, 48.79]}},
                ]
            },
        )

    result = places.search_nearby_places(
        context_with(demo, handler, session_location=HOME), "gym"
    )

    assert [place.name for place in result.places] == ["Near gym", "Far gym"]
    assert result.places[0].map_url == "https://www.openstreetmap.org/node/42"
    assert result.places[0].distance_km < result.places[1].distance_km
    assert "OpenStreetMap" in result.data_source


def test_every_tag_of_the_supplement_category_is_searched(demo):
    tags = []

    def handler(request):
        tags.append(request.url.params["osm_tag"])
        return httpx.Response(200, json={"features": []})

    places.search_nearby_places(
        context_with(demo, handler, session_location=HOME), "supplement_store"
    )

    assert (
        tags == places.places_settings()["categories"]["supplement_store"]["osm_tags"]
    )


def test_a_fitness_name_hint_comes_from_the_configuration(demo):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "features": [
                    feature("Decathlon City", 48.791),
                    feature("Fencing Shop", 48.792),
                ]
            },
        )

    result = places.search_nearby_places(
        context_with(demo, handler, session_location=HOME), "sports_store"
    )

    hints = {
        place.name: place.name_suggests_fitness_equipment for place in result.places
    }
    assert hints == {"Decathlon City": True, "Fencing Shop": False}


@pytest.mark.parametrize(
    "response", [httpx.Response(503), httpx.Response(200, text="not json")]
)
def test_a_map_service_failure_is_reported(demo, response):
    result = places.search_nearby_places(
        context_with(demo, lambda r: response, session_location=HOME), "gym"
    )

    assert result.code == "places_unavailable"


def test_a_radius_outside_the_configured_limits_is_refused(demo):
    result = places.search_nearby_places(
        context_with(demo, lambda r: httpx.Response(200), session_location=HOME),
        "gym",
        radius_km=500,
    )

    assert result.code == "invalid_request"


# --- service area (metropolitan France) and unrelated places -----------------


def test_only_the_best_match_counts(demo):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "features": [
                    feature("Bruxelles", 50.85, 4.35, countrycode="BE", type="city"),
                    feature("Bruxelles", 47.77, 2.88, countrycode="FR", type="other"),
                ]
            },
        )

    result = places.get_location(context_with(demo, handler), "Bruxelles")

    assert result.code == "outside_service_area"
    assert "postcode" in result.error


@pytest.mark.parametrize("kind", ["country", "state"])
def test_a_country_or_region_is_too_vague(demo, kind):
    found = feature("France", 46.6, 2.4, countrycode="FR", type=kind)
    result = places.get_location(
        context_with(demo, lambda r: httpx.Response(200, json={"features": [found]})),
        "France",
    )

    assert result.code == "place_too_vague"


def test_a_place_that_only_exists_abroad_is_refused(demo):
    berlin = feature("Berlin", 52.52, 13.4, countrycode="DE", type="city")
    result = places.get_location(
        context_with(demo, lambda r: httpx.Response(200, json={"features": [berlin]})),
        "Berlin",
    )

    assert result.code == "outside_service_area"


def test_shared_or_saved_coordinates_outside_france_are_refused(demo):
    abroad = Location(label="New York", latitude=40.7, longitude=-74.0)
    shared = places.search_nearby_places(
        context_with(demo, lambda r: httpx.Response(500), session_location=abroad),
        "gym",
    )
    demo.repository.save_location(demo.student_id, "London", 51.5, -0.12)
    saved = places.get_location(context_with(demo, lambda r: httpx.Response(500)))

    assert shared.code == saved.code == "outside_service_area"


def test_unrelated_places_are_dropped_by_whole_word(demo):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "features": [
                    feature("Escrime Diffusion", 48.791),
                    feature("Running Conseil", 48.792),
                    feature("Brunch Fitness Store", 48.793),
                    feature("Décathlon Cachan", 48.794),
                ]
            },
        )

    result = places.search_nearby_places(
        context_with(demo, handler, session_location=HOME), "sports_store"
    )

    assert [place.name for place in result.places] == [
        "Brunch Fitness Store",
        "Décathlon Cachan",
    ]
    assert all(place.name_suggests_fitness_equipment for place in result.places)


# --- search_documents -----------------------------------------------------------


class FakeRetriever:
    def __init__(self, error=None, chunks=1):
        self.error, self.chunks = error, chunks

    def search(self, _query, _top_k):
        if self.error:
            raise self.error
        return [
            RetrievedChunk(
                text="x" * 1000,
                chunk_id=f"c{i}",
                video_id="v",
                passage_ids=["p"],
                title="Rest times",
                video_url="https://example.com/v",
            )
            for i in range(self.chunks)
        ]


class Gated:
    """Like QueryExpansionRetriever: it refuses non-plan requests, but wraps a search."""

    def __init__(self, inner):
        self.retriever = inner

    def search(self, _query, _top_k):
        raise ScopeRefusalError("out_of_scope", "Plan requests only.")


def search_context(demo, retriever):
    pipeline = type(
        "Pipeline", (), {"retriever": Gated(retriever), "generator": None}
    )()
    return make_context(
        demo.repository, demo.student_id, pipeline_factory=lambda: pipeline
    )


def test_search_skips_the_plan_scope_gate_and_cites_passages(demo):
    result = knowledge.search_documents(
        search_context(demo, FakeRetriever()), "rest between sets"
    )

    assert result.evidence_found
    assert (result.documents[0].source_id, result.documents[0].title) == (
        "c0",
        "Rest times",
    )
    assert len(result.documents[0].excerpt) == knowledge.EXCERPT_CHARACTERS


def test_no_evidence_is_said_plainly(demo):
    result = knowledge.search_documents(
        search_context(demo, FakeRetriever(chunks=0)), "underwater basket weaving"
    )

    assert not result.evidence_found and result.documents == []


@pytest.mark.parametrize(
    "query", ["My knee hurts, what should I do?", "J'ai mal au dos, que faire ?"]
)
def test_health_questions_are_refused_before_searching(demo, query):
    context = search_context(demo, FakeRetriever(AssertionError("must not search")))

    assert knowledge.search_documents(context, query).code == "health_question"


def test_search_failures_are_reported(demo):
    assert (
        knowledge.search_documents(
            search_context(demo, FakeRetriever(RuntimeError("down"))), "x y z"
        ).code
        == "search_unavailable"
    )
    assert (
        knowledge.search_documents(demo.context, "rest times").code
        == "search_unavailable"
    )
