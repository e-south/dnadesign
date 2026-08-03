"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_badge_picture_sources.py

Tests rendered picture-source selection for documentation badge detection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from dnadesign.devtools.docs.badges import rendered_markdown_badge_lines


@pytest.mark.parametrize(
    "content",
    [
        ('<picture><source srcset="diagram.svg"><source srcset="badge.svg"><img src="fallback.svg"></picture>\n'),
        '<picture><source srcset="diagram.svg"><img src="badge.svg"></picture>\n',
        (
            '<picture><source srcset="diagram.svg 320w" sizes="invalid ???">'
            '<source srcset="badge.svg"><img src="fallback.svg"></picture>\n'
        ),
        (
            '<picture><source srcset="https://example.test:bogus/diagram.svg">'
            '<source srcset="badge.svg"><img src="fallback.svg"></picture>\n'
        ),
        '<picture><img src="diagram.svg"><source srcset="badge.svg"></picture>\n',
    ],
)
def test_unconditional_picture_source_hides_unreachable_badge_urls(content: str) -> None:
    assert rendered_markdown_badge_lines(content) == ()


@pytest.mark.parametrize("media", ["", "   ", "all", " ALL ", "/**/all/**/"])
def test_unconditional_picture_media_hides_later_badge_sources(media: str) -> None:
    content = (
        f'<picture><source media="{media}" srcset="diagram.svg">'
        '<source srcset="badge.svg"><img src="fallback.svg"></picture>\n'
    )

    assert rendered_markdown_badge_lines(content) == ()


@pytest.mark.parametrize(
    "content",
    [
        '<picture><source srcset="badge.svg"><img src="fallback.svg"></picture>\n',
        ('<picture><source srcset=""><source srcset="badge.svg"><img src="fallback.svg"></picture>\n'),
        ('<picture><source srcset="diagram.svg 1.x"><source srcset="badge.svg"><img src="fallback.svg"></picture>\n'),
        (
            '<picture><source media="screen" srcset="diagram.svg"><source srcset="badge.svg">'
            '<img src="fallback.svg"></picture>\n'
        ),
        (
            '<picture><source media="invalid ???" srcset="diagram.svg"><source srcset="badge.svg">'
            '<img src="fallback.svg"></picture>\n'
        ),
        ('<picture><source srcset="diagram.svg 1x, badge.svg 2x"><img src="fallback.svg"></picture>\n'),
        (
            '<a href="report"><picture><source srcset="diagram.svg">'
            '<img alt="Coverage" src="fallback.svg"></picture></a>\n'
        ),
    ],
)
def test_reachable_picture_badges_are_detected(content: str) -> None:
    assert rendered_markdown_badge_lines(content) == (content.strip(),)


@pytest.mark.parametrize("declared_type", ["", "text/plain", "image/svg+xml", "invalid"])
def test_declared_picture_source_type_keeps_later_sources_inspectable(declared_type: str) -> None:
    content = (
        f'<picture><source type="{declared_type}" srcset="diagram.svg">'
        '<source srcset="badge.svg"><img src="fallback.svg"></picture>\n'
    )

    assert rendered_markdown_badge_lines(content) == (content.strip(),)


def test_picture_source_reachability_is_computed_per_image() -> None:
    content = (
        "<picture>\n"
        '<source media="screen" srcset="diagram.svg">\n'
        '<img src="badge.svg">\n'
        '<source srcset="diagram-2.svg">\n'
        '<img src="badge.svg">\n'
        "</picture>\n"
    )

    assert rendered_markdown_badge_lines(content) == ('<img src="badge.svg">',)
