"""Password hashing and bearer-token helpers for the local demo API."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def hash_password(password: str) -> str:
    """Hash a password with the recommended Argon2 configuration.

    Args:
        password: Plain-text password to hash.

    Returns:
        An encoded password hash suitable for storage.

    Raises:
        RuntimeError: If the ``pwdlib[argon2]`` dependency is unavailable.
    """
    try:
        from pwdlib import PasswordHash
    except ImportError as error:
        raise RuntimeError(
            "Authentication needs pwdlib[argon2]. Install requirements.txt."
        ) from error
    return PasswordHash.recommended().hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a login attempt against a stored password hash.

    Args:
        password: Plain-text password supplied for the login attempt.
        password_hash: Encoded hash previously returned by :func:`hash_password`.

    Returns:
        Whether the password matches the stored hash.
    """
    from pwdlib import PasswordHash

    return PasswordHash.recommended().verify(password, password_hash)


def create_access_token(user_id: str, secret: str, lifetime_minutes: int = 60) -> str:
    """Sign a JWT containing an account ID and expiration time.

    Args:
        user_id: Internal account ID to place in the token subject.
        secret: Application secret used to sign the token with HS256.
        lifetime_minutes: Token lifetime from the current UTC time.

    Returns:
        The encoded JWT access token.
    """
    import jwt

    expires_at = datetime.now(UTC) + timedelta(minutes=lifetime_minutes)
    return jwt.encode({"sub": user_id, "exp": expires_at}, secret, algorithm="HS256")


def read_access_token(token: str, secret: str) -> str:
    """Decode a signed token and return its internal account ID.

    Args:
        token: Encoded bearer token to verify.
        secret: Application secret used to verify its HS256 signature.

    Returns:
        The non-empty string subject identifying the account.

    Raises:
        jwt.InvalidTokenError: If the signature, claims, or expiration is invalid.
        ValueError: If the token has no non-empty string subject.
    """
    import jwt

    payload = jwt.decode(token, secret, algorithms=["HS256"])
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("Access token has no account identity.")
    return user_id
