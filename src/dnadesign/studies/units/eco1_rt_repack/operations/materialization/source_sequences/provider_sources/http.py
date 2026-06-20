"""HTTP helpers for Eco1 provider-source acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.request import Request, urlopen

_USER_AGENT = "dnadesign-eco1-rt-repack/1.0"


def fetch_text(url: str, *, headers: Mapping[str, str] | None = None, timeout_seconds: float = 60.0) -> str:
    """Fetch UTF-8 text from a provider endpoint."""

    request_headers = {"User-Agent": _USER_AGENT}
    request_headers.update(headers or {})
    request = Request(url, headers=request_headers)
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - declared provider endpoints only.
        return response.read().decode("utf-8")
