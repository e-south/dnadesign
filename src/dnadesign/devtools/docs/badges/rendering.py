"""Render Markdown while retaining raw HTML and image source locations."""

from __future__ import annotations

import secrets
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from markdown_it import MarkdownIt

from dnadesign.devtools.docs.parser_compat import (
    MarkdownStateInline,
    MarkdownToken,
    markdown_html_inline_rule,
    markdown_image_rule,
)

INLINE_SOURCE_SPAN_META = "dnadesign_source_span"


@dataclass(frozen=True, slots=True)
class RawHTMLFragment:
    content: str
    source_start_line: int


@dataclass(frozen=True, slots=True)
class RawHTMLSourceSpan:
    rendered_start: int
    rendered_end: int
    rendered_start_line: int
    source_start_line: int


@dataclass(frozen=True, slots=True)
class RawHTMLSourceMap:
    spans: tuple[RawHTMLSourceSpan, ...]
    rendered_starts: tuple[int, ...]

    @classmethod
    def from_fragments(
        cls,
        rendered_html: str,
        fragments: Sequence[RawHTMLFragment],
    ) -> RawHTMLSourceMap:
        line_starts = _line_start_offsets(rendered_html)
        spans: list[RawHTMLSourceSpan] = []
        search_start = 0
        for fragment in fragments:
            if not fragment.content or fragment.source_start_line < 1:
                raise RuntimeError("raw HTML source fragment has an invalid source contract")
            rendered_start = rendered_html.find(fragment.content, search_start)
            if rendered_start < 0:
                raise RuntimeError("rendered Markdown omitted a declared raw HTML source fragment")
            rendered_end = rendered_start + len(fragment.content)
            spans.append(
                RawHTMLSourceSpan(
                    rendered_start=rendered_start,
                    rendered_end=rendered_end,
                    rendered_start_line=bisect_right(line_starts, rendered_start),
                    source_start_line=fragment.source_start_line,
                )
            )
            search_start = rendered_end
        return cls(
            spans=tuple(spans),
            rendered_starts=tuple(span.rendered_start for span in spans),
        )

    def source_line_for(self, *, rendered_offset: int, rendered_line: int) -> int | None:
        span_index = bisect_right(self.rendered_starts, rendered_offset) - 1
        if span_index < 0:
            return None
        span = self.spans[span_index]
        if rendered_offset >= span.rendered_end:
            return None
        line_offset = rendered_line - span.rendered_start_line
        if line_offset < 0:
            raise RuntimeError("HTML tokenizer resolved a tag before its raw source fragment")
        return span.source_start_line + line_offset


def _line_start_offsets(content: str) -> tuple[int, ...]:
    return (0, *(index + 1 for index, character in enumerate(content) if character == "\n"))


def _record_inline_source_span(
    state: MarkdownStateInline,
    silent: bool,
    rule: Callable[[MarkdownStateInline, bool], bool],
    *,
    expected_type: str,
) -> bool:
    start = state.pos
    token_count = len(state.tokens)
    matched = rule(state, silent)
    if not matched or silent:
        return matched
    produced_tokens = state.tokens[token_count:]
    if not produced_tokens or produced_tokens[-1].type != expected_type:
        raise RuntimeError(f"source-mapped inline rule did not emit its expected {expected_type} token")
    produced_tokens[-1].meta[INLINE_SOURCE_SPAN_META] = (start, state.pos)
    return True


def _source_mapped_image_rule(state: MarkdownStateInline, silent: bool) -> bool:
    return _record_inline_source_span(state, silent, markdown_image_rule, expected_type="image")


def _source_mapped_html_inline_rule(state: MarkdownStateInline, silent: bool) -> bool:
    return _record_inline_source_span(state, silent, markdown_html_inline_rule, expected_type="html_inline")


MARKDOWN = MarkdownIt("commonmark")
MARKDOWN.inline.ruler.at("image", _source_mapped_image_rule)
MARKDOWN.inline.ruler.at("html_inline", _source_mapped_html_inline_rule)


def _annotate_inline_candidates(
    children: Sequence[MarkdownToken],
    *,
    source: str,
    start_line_no: int,
    marker_attribute: str,
) -> tuple[RawHTMLFragment, ...]:
    raw_fragments: list[RawHTMLFragment] = []
    line_cursor = 0
    line_offset = 0
    for token in children:
        if token.type not in {"html_inline", "image"}:
            continue
        span = token.meta.get(INLINE_SOURCE_SPAN_META)
        if not isinstance(span, tuple) or len(span) != 2 or not all(isinstance(position, int) for position in span):
            raise RuntimeError(f"{token.type} token is missing its source-span contract")
        span_start, span_end = span
        if span_start < line_cursor or span_end < span_start or span_end > len(source):
            raise RuntimeError(f"{token.type} token has an invalid source span")
        line_offset += source[line_cursor:span_start].count("\n")
        line_cursor = span_start
        token_line_no = start_line_no + line_offset
        if token.type == "image":
            token.attrSet(marker_attribute, str(token_line_no))
        else:
            if token.content != source[span_start:span_end]:
                raise RuntimeError("html_inline token content differs from its declared source span")
            raw_fragments.append(
                RawHTMLFragment(
                    content=token.content,
                    source_start_line=token_line_no,
                )
            )
    return tuple(raw_fragments)


def render_markdown_with_markers(content: str) -> tuple[str, str, RawHTMLSourceMap]:
    """Render Markdown and return the HTML, marker name, and raw-source map."""

    environment: dict[str, object] = {}
    tokens = MARKDOWN.parse(content, environment)
    marker_attribute = f"data-dnadesign-image-{secrets.token_hex(12)}"
    raw_fragments: list[RawHTMLFragment] = []
    for token in tokens:
        if token.type == "inline" and token.map is not None:
            raw_fragments.extend(
                _annotate_inline_candidates(
                    token.children or (),
                    source=token.content,
                    start_line_no=token.map[0] + 1,
                    marker_attribute=marker_attribute,
                )
            )
        elif token.type == "html_block" and token.map is not None:
            raw_fragments.append(
                RawHTMLFragment(
                    content=token.content,
                    source_start_line=token.map[0] + 1,
                )
            )
    rendered_html = MARKDOWN.renderer.render(tokens, MARKDOWN.options, environment)
    return (
        rendered_html,
        marker_attribute,
        RawHTMLSourceMap.from_fragments(rendered_html, raw_fragments),
    )
