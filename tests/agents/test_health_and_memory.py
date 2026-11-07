"""The health check (fail closed) and the agent's semantic and episodic memory."""

from datetime import timedelta

from src.agents import memory
from src.agents.health import OllamaHealthChecker
from src.tools.context import utc_now
from tests.conftest import make_context


class FakeOllama:
    """Answers like ``ollama.Client.chat``, or raises."""

    def __init__(self, content=None, error=None):
        self.content, self.error, self.calls = content, error, 0

    def chat(self, **_arguments):
        self.calls += 1
        if self.error:
            raise self.error
        return {"message": {"content": self.content}}


def checker_with(client):
    checker = OllamaHealthChecker("tiny-model", "http://ollama.test")
    checker._client = client
    return checker


# --- health check --------------------------------------------------------------


def test_the_model_decision_is_used_and_cached():
    client = FakeOllama('{"health_concern": true, "reason": "knee pain"}')
    checker = checker_with(client)

    first = checker.check("Mon genou a gonflé après les squats")
    second = checker.check("Mon genou a gonflé après les squats")

    assert first.health_concern and second.health_concern
    assert client.calls == 1


def test_normal_training_effort_is_not_a_concern():
    checker = checker_with(FakeOllama('{"health_concern": false, "reason": "effort"}'))

    assert not checker.check("The last set was really hard").health_concern


def test_empty_text_never_calls_the_model():
    client = FakeOllama(error=AssertionError("must not be called"))

    assert not checker_with(client).check("   ").health_concern
    assert not checker_with(client).check(None).health_concern


def test_an_unreachable_model_fails_closed_and_is_retried_later():
    client = FakeOllama(error=ConnectionError("Ollama is down"))
    checker = checker_with(client)

    assert checker.check("Everything was fine").health_concern
    client.error, client.content = None, '{"health_concern": false, "reason": "fine"}'
    assert not checker.check("Everything was fine").health_concern


def test_a_malformed_answer_fails_closed():
    assert checker_with(FakeOllama("not json")).check("ok").health_concern
    assert checker_with(FakeOllama('{"unexpected": 1}')).check("ok").health_concern


# --- semantic memory -------------------------------------------------------------


def test_a_remembered_fact_is_recalled_in_a_new_context(demo):
    assert memory.remember_fact(demo.context, "I prefer morning sessions").ok

    fresh = make_context(demo.repository, demo.student_id)
    assert "I prefer morning sessions" in memory.recall(fresh)


def test_saving_a_fact_twice_keeps_one_copy(demo):
    memory.remember_fact(demo.context, "I prefer mornings")
    memory.remember_fact(demo.context, "I prefer mornings")

    assert demo.repository.list_memory_facts(demo.student_id, 10) == [
        "I prefer mornings"
    ]


def test_health_details_are_never_remembered(demo):
    result = memory.remember_fact(demo.context, "I have knee pain since last week")

    assert result.code == "health_not_stored"
    assert demo.repository.list_memory_facts(demo.student_id, 10) == []


def test_memory_is_private_to_each_student(demo, other_student):
    memory.remember_fact(demo.context, "I prefer mornings")

    assert memory.recall(make_context(demo.repository, other_student)) == ""


# --- episodic memory -------------------------------------------------------------


def test_an_episode_hides_a_health_related_request():
    episode = memory.describe_episode(
        "My back hurts, adapt my plan", ["Planning"], [], [], health_concern=True
    )

    assert "hurts" not in episode
    assert memory.HEALTH_REQUEST_PLACEHOLDER in episode


def test_episodes_older_than_the_retention_period_are_dropped(demo):
    # The database stamps rows with the real time, so use the real clock here.
    context = make_context(demo.repository, demo.student_id, clock=utc_now)
    demo.repository.save_memory_episode(demo.student_id, "Very old task")
    with demo.repository._connect() as connection:
        connection.execute(
            "UPDATE memory_episodes SET created_at=? WHERE episode='Very old task'",
            ((utc_now() - timedelta(days=200)).isoformat(),),
        )

    memory.save_episode(context, "Recent task")

    recalled = memory.recall(context)
    assert "Recent task" in recalled and "Very old task" not in recalled
    assert demo.repository.list_memory_episodes(demo.student_id, 10) == ["Recent task"]
