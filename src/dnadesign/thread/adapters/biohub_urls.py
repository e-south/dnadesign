"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/biohub_urls.py

Shared Biohub endpoint validation for thread adapters.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from urllib.parse import urlsplit

TRUSTED_BIOHUB_API_HOSTS = frozenset({"biohub.ai", "www.biohub.ai"})


def validate_biohub_api_base_url(base_url: str, *, service_label: str = "Biohub API") -> str:
    """Return a normalized Biohub base URL after enforcing the public endpoint."""

    parsed = urlsplit(str(base_url))
    host = parsed.hostname.lower() if parsed.hostname else ""
    if (
        parsed.scheme != "https"
        or host not in TRUSTED_BIOHUB_API_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.path not in {"", "/"}
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        message = f"{service_label} base URL must be https://biohub.ai or https://www.biohub.ai"
        raise ValueError(message)
    return f"https://{host}"
