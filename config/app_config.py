"""Configuration for the Streamlit presentation layer."""

from __future__ import annotations

import os

from dotenv import load_dotenv


def get_api_url() -> str:
    """Return the backend URL configured for the Streamlit client.

    The value is read from ``REPWISE_API_URL`` after loading the local dotenv
    file, defaults to ``http://127.0.0.1:8000``, and has trailing slashes
    removed.

    Returns:
        Base URL used by the Streamlit client for backend requests.
    """
    load_dotenv()
    return os.getenv("REPWISE_API_URL", "http://127.0.0.1:8000").rstrip("/")
