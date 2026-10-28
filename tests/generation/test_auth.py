"""Exercise the real password hashing and signed-token helpers when installed."""

import pytest


def test_passwords_are_hashed_and_tokens_expire():
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
