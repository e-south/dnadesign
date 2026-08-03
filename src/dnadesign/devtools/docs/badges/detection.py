"""Classify image URLs and labels that represent documentation badges."""

from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import unquote

from upa_url import URL

BADGE_PATH_PATTERN = re.compile(r"(?:^|[/_.-])badges?(?:[./?_-]|$)", flags=re.IGNORECASE)
BADGE_PROVIDER_HOSTS = frozenset({"shields.io", "codecov.io"})
BADGE_LABEL_PATTERN = re.compile(r"\s*(?:ci|coverage|codecov|license)\s*", flags=re.IGNORECASE)
MARKDOWN_RENDER_BASE_URL = "https://example.test/docs/"


def source_has_badge_hint(source: str) -> bool:
    """Return whether an image source has a recognized badge signal."""

    if any("\ud800" <= character <= "\udfff" for character in source):
        return False
    parsed_source = URL.parse(source, MARKDOWN_RENDER_BASE_URL)
    if parsed_source is None:
        return False
    if _is_badge_provider_hostname(parsed_source.hostname):
        return True
    path = unquote(parsed_source.pathname)
    return BADGE_PATH_PATTERN.search(path) is not None


def _is_badge_provider_hostname(hostname: str) -> bool:
    normalized_hostname = hostname[:-1] if hostname.endswith(".") else hostname
    return any(
        normalized_hostname == provider or normalized_hostname.endswith(f".{provider}")
        for provider in BADGE_PROVIDER_HOSTS
    )


def looks_like_badge(*, label: str, sources: Sequence[str], linked: bool) -> bool:
    """Return whether a rendered image should be governed as a badge."""

    return any(source_has_badge_hint(source) for source in sources) or (
        linked and BADGE_LABEL_PATTERN.fullmatch(label) is not None
    )
