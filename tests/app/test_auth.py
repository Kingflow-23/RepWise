"""Exercise the real password hashing and signed-token helpers when installed."""

import pytest

from fastapi.testclient import TestClient

from config.config import AppConfig
from src.api.main import create_app
from src.api.auth import create_access_token, hash_password
from src.db.repository import StudentRepository
from src.retrieval.base import MissingRetriever


@pytest.fixture
def account_api(tmp_path):
    secret = "test-secret-with-at-least-32-bytes-for-hmac"
    repository = StudentRepository(tmp_path / "auth.sqlite3")
    repository.initialize()
    user_id = repository.create_user("Alex Test", hash_password("TestPassword123!"))
    app = create_app(
        AppConfig(database_path=repository.database_path, auth_secret=secret),
        repository=repository,
        retriever=MissingRetriever("Not needed for authentication"),
    )
    return TestClient(app), user_id, secret


def test_current_account_returns_only_verified_public_identity(account_api):
    client, user_id, _ = account_api
    login = client.post(
        "/api/v1/auth/login",
        json={"full_name": "Alex Test", "password": "TestPassword123!"},
    )
    assert login.status_code == 200
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )
    assert response.status_code == 200
    assert response.json() == {"user_id": user_id, "full_name": "Alex Test"}


def test_current_account_requires_authentication(account_api):
    client, _, _ = account_api
    assert client.get("/api/v1/auth/me").status_code == 401
    assert (
        client.get(
            "/api/v1/auth/me", headers={"Authorization": "Bearer invalid"}
        ).status_code
        == 401
    )


def test_current_account_rejects_expired_and_unknown_accounts(account_api):
    client, user_id, secret = account_api
    for token in (
        create_access_token(user_id, secret, lifetime_minutes=-1),
        create_access_token("no-longer-exists", secret),
    ):
        response = client.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401


def test_passwords_are_hashed_and_tokens_expire():
    """Verify password helpers and signed-token validation when dependencies exist."""
    pytest.importorskip("pwdlib")
    pytest.importorskip("jwt")

    from src.api.auth import (
        create_access_token,
        hash_password,
        read_access_token,
        verify_password,
    )

    saved_hash = hash_password("correct horse battery staple")
    assert saved_hash != "correct horse battery staple"
    assert verify_password("correct horse battery staple", saved_hash)
    assert not verify_password("wrong password", saved_hash)

    test_secret = "test-secret-with-at-least-32-bytes-for-hmac"
    token = create_access_token("student-a", test_secret)
    assert read_access_token(token, test_secret) == "student-a"
    with pytest.raises(Exception):
        read_access_token(token, "different-secret-with-at-least-32-bytes")
