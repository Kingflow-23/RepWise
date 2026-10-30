"""Password hashing and bearer-token helpers for the local demo API."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def hash_password(password: str) -> str:
    """Store an Argon2 password hash, never the supplied password."""
    try:
        from pwdlib import PasswordHash
    except ImportError as error:
        raise RuntimeError(
            "Authentication needs pwdlib[argon2]. Install requirements.txt."
        ) from error
    return PasswordHash.recommended().hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Check a login attempt with the same password-hashing library."""
    from pwdlib import PasswordHash

    return PasswordHash.recommended().verify(password, password_hash)


def create_access_token(user_id: str, secret: str, lifetime_minutes: int = 60) -> str:
    """Sign a short-lived JWT access token using the configured application secret."""
    import jwt

    expires_at = datetime.now(UTC) + timedelta(minutes=lifetime_minutes)
    return jwt.encode({"sub": user_id, "exp": expires_at}, secret, algorithm="HS256")


def read_access_token(token: str, secret: str) -> str:
    """Return the internal user ID from a valid, unexpired access token."""
    import jwt

    payload = jwt.decode(token, secret, algorithms=["HS256"])
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("Access token has no account identity.")
    return user_id
