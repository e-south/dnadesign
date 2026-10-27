"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_badge_urls.py

Tests browser URL contracts for documentation badge detection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from dnadesign.devtools.docs.badges.detection import source_has_badge_hint


@pytest.mark.parametrize(
    "source",
    [
        "https://💩.shields.io/status.svg",
        "https://ci_internal.img.shields.io/status.svg",
        "https://[2001:db8::1]/badge.svg",
        "https://[::ffff:192.0.2.1]/badge.svg",
        "https://127.1/badge.svg",
        "https://%31%32%37%2E%30%2E%30%2E%31/badge.svg",
        "https://ci_internal.test/badge.svg",
        "https://gitlab.example/group/project/-/badges/main/pipeline.svg",
        "file://[2001:db8::1]/badge.svg",
        "https:badge.svg",
        "https:/assets/badge.svg",
        r"https:\assets\badge.svg",
        r"file://ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ/status.svg",
        r"file:\\ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ\status.svg",
        r"file:/\ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ/status.svg",
        r"file:\/ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ\status.svg",
    ],
)
def test_browser_valid_badge_sources_are_detected(source: str) -> None:
    assert source_has_badge_hint(source)


@pytest.mark.parametrize(
    "source",
    [
        r"https:img.shields.io/status.svg",
        r"https:ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ/status.svg",
        r"https:/ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ/status.svg",
        r"https:\ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ\status.svg",
        r"https:\img%E3%80%82shields%E3%80%82io\status.svg",
        "https://img.shields.io.example.test/status.svg",
        "https://example.test/img.shields.io/status.svg",
        r"file:ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ/status.svg",
        r"file:/ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ/status.svg",
        r"file:\ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ\status.svg",
        r"file:///ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ/status.svg",
        r"file:\\\ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ\status.svg",
        r"file:/\/ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ/status.svg",
    ],
)
def test_browser_valid_non_badge_sources_are_ignored(source: str) -> None:
    assert not source_has_badge_hint(source)


@pytest.mark.parametrize(
    "source",
    [
        "https://xn--.shields.io/status.svg",
        "https://xn--abc.shields.io/status.svg",
        "https://foo＼img.shields.io/status.svg",
        "https://foo［bar］.img.shields.io/status.svg",
        "https://foo％img.shields.io/status.svg",
        "https://%3A%3A1/badge.svg",
        "https://[v1.foo]/badge.svg",
        "https://[::1%25eth0]/badge.svg",
        "https://999.999.999.999/badge.svg",
        "//256.0.0.1/badge.svg",
        "https://09/badge.svg",
        "file://[v1.foo]/badge.svg",
        "file://img.shields.io:/badge.svg",
        "file://img.shields.io:80/badge.svg",
        "file://user@img.shields.io/badge.svg",
        "file://user" + ":" + "credential@img.shields.io/badge.svg",
        "file://@img.shields.io/badge.svg",
        "https://example.test/\ud800/badge.svg",
    ],
)
def test_browser_invalid_sources_are_ignored(source: str) -> None:
    assert not source_has_badge_hint(source)


@pytest.mark.parametrize("authority", ["//ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ", "https://img.shields.io"])
@pytest.mark.parametrize("port", ["", "0", "1", "80", "443", "65535"])
def test_browser_valid_provider_ports_are_detected(authority: str, port: str) -> None:
    assert source_has_badge_hint(f"{authority}:{port}/status.svg")


@pytest.mark.parametrize("authority", ["//ｉｍｇ．ｓｈｉｅｌｄｓ．ｉｏ", "https://img.shields.io"])
@pytest.mark.parametrize("port", ["bogus", "65536", "99999", "%34%34%33", "%38%30", "%2F", "%3F", "%23", "%40"])
def test_browser_invalid_ports_are_ignored(authority: str, port: str) -> None:
    assert not source_has_badge_hint(f"{authority}:{port}/badge.svg")


@pytest.mark.parametrize("authority_prefix", ["//", "https://"])
@pytest.mark.parametrize("encoded_delimiter", ["%3A443", "%2F", "%3F", "%23", "%40"])
def test_browser_invalid_encoded_host_delimiters_are_ignored(
    authority_prefix: str,
    encoded_delimiter: str,
) -> None:
    source = f"{authority_prefix}img.shields.io{encoded_delimiter}/badge.svg"
    assert not source_has_badge_hint(source)
