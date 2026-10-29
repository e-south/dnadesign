"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/parser_compat.py

Contains the bounded private parser hooks used by documentation checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from itertools import count
from typing import Protocol

import html5lib
from html5lib._tokenizer import HTMLTokenizer
from html5lib._tokenizer import tokenTypes as token_types
from html5lib.html5parser import HTMLParser
from html5lib.html5parser import _ReparseException as ReparseException
from markdown_it.rules_inline import html_inline as markdown_html_inline_rule
from markdown_it.rules_inline import image as markdown_image_rule
from markdown_it.rules_inline.state_inline import StateInline as MarkdownStateInline
from markdown_it.token import Token as MarkdownToken


class SourceLineMap(Protocol):
    """Map rendered HTML offsets back to Markdown source lines."""

    def source_line_for(self, *, rendered_offset: int, rendered_line: int) -> int | None: ...


def _line_start_offsets(content: str) -> tuple[int, ...]:
    return (0, *(index + 1 for index, character in enumerate(content) if character == "\n"))


class _SourceMappedHTMLTokenizer(HTMLTokenizer):
    def __init__(
        self,
        stream: str,
        *,
        parser: HTMLParser,
        marker_attribute: str,
        raw_source_map: SourceLineMap,
        **kwargs: object,
    ) -> None:
        if not isinstance(stream, str):
            raise TypeError("source-mapped HTML parsing requires rendered text")
        self._line_starts = _line_start_offsets(stream)
        super().__init__(stream, parser=parser, **kwargs)
        self._marker_attribute = marker_attribute
        self._raw_source_map = raw_source_map
        self._ordinals = count()

    def tagOpenState(self) -> bool:
        rendered_line, rendered_column = self.stream.position()
        rendered_offset = self._line_starts[rendered_line - 1] + rendered_column - 1
        previous_token = getattr(self, "currentToken", None)
        result = super().tagOpenState()
        token = getattr(self, "currentToken", None)
        if token is not previous_token and token is not None and token.get("type") == token_types["StartTag"]:
            token["dnadesign_rendered_line"] = rendered_line
            token["dnadesign_rendered_offset"] = rendered_offset
        return result

    def emitCurrentToken(self) -> None:
        token = self.currentToken
        rendered_line = token.pop("dnadesign_rendered_line", None)
        rendered_offset = token.pop("dnadesign_rendered_offset", None)
        super().emitCurrentToken()
        if token.get("type") != token_types["StartTag"] or token.get("name") not in {"image", "img", "source"}:
            return
        attributes = token.get("data")
        if not isinstance(attributes, dict):
            raise RuntimeError("HTML tokenizer emitted a start tag without normalized attributes")
        source_line = None
        if isinstance(rendered_line, int) and isinstance(rendered_offset, int):
            source_line = self._raw_source_map.source_line_for(
                rendered_offset=rendered_offset,
                rendered_line=rendered_line,
            )
        if source_line is None:
            declared_line = attributes.get(self._marker_attribute)
            if isinstance(declared_line, str) and declared_line.isdigit():
                source_line = int(declared_line)
        if source_line is None:
            return
        if source_line < 1:
            raise RuntimeError("HTML tokenizer resolved an invalid source line")
        attributes[self._marker_attribute] = f"{source_line}:{next(self._ordinals)}"


class SourceMappedHTMLParser(HTMLParser):
    """html5lib parser that retains Markdown source locations for images."""

    def __init__(self, *, marker_attribute: str, raw_source_map: SourceLineMap) -> None:
        self._marker_attribute = marker_attribute
        self._raw_source_map = raw_source_map
        super().__init__(tree=html5lib.getTreeBuilder("etree"), namespaceHTMLElements=True)

    def _parse(
        self,
        stream: str,
        innerHTML: bool = False,
        container: str = "div",
        scripting: bool = False,
        **kwargs: object,
    ) -> None:
        self.innerHTMLMode = innerHTML
        self.container = container
        self.scripting = scripting
        self.tokenizer = _SourceMappedHTMLTokenizer(
            stream,
            parser=self,
            marker_attribute=self._marker_attribute,
            raw_source_map=self._raw_source_map,
            **kwargs,
        )
        self.reset()
        try:
            self.mainLoop()
        except ReparseException:
            self.reset()
            self.mainLoop()


__all__ = [
    "MarkdownStateInline",
    "MarkdownToken",
    "SourceMappedHTMLParser",
    "markdown_html_inline_rule",
    "markdown_image_rule",
]
