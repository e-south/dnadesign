"""Resolve images that survive Markdown rendering and HTML selection rules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from xml.etree.ElementTree import Element

from dnadesign.devtools.docs.badges.rendering import RawHTMLSourceMap, render_markdown_with_markers
from dnadesign.devtools.docs.badges.responsive import (
    html_image_sources,
    picture_source_is_potentially_eligible,
    picture_source_is_terminal,
    selectable_srcset_urls,
)
from dnadesign.devtools.docs.parser_compat import SourceMappedHTMLParser

XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_HREF_ATTRIBUTE = "{http://www.w3.org/1999/xlink}href"
NON_RENDERING_CONTAINERS = frozenset(
    {
        (XHTML_NAMESPACE, "template"),
        (SVG_NAMESPACE, "clippath"),
        (SVG_NAMESPACE, "desc"),
        (SVG_NAMESPACE, "defs"),
        (SVG_NAMESPACE, "marker"),
        (SVG_NAMESPACE, "mask"),
        (SVG_NAMESPACE, "metadata"),
        (SVG_NAMESPACE, "pattern"),
        (SVG_NAMESPACE, "script"),
        (SVG_NAMESPACE, "style"),
        (SVG_NAMESPACE, "symbol"),
        (SVG_NAMESPACE, "title"),
    }
)


@dataclass(frozen=True, slots=True)
class ImageSpec:
    label: str
    sources: tuple[str, ...]
    linked: bool


@dataclass(frozen=True, slots=True)
class ImageOccurrence:
    line_no: int
    ordinal: int
    spec: ImageSpec


@dataclass(frozen=True, slots=True)
class RenderedMarkdownImage:
    """A rendered image and its source location in Markdown."""

    line_no: int
    label: str
    sources: tuple[str, ...]
    linked: bool


def _element_name(element: Element) -> tuple[str, str] | None:
    if not isinstance(element.tag, str) or not element.tag.startswith("{"):
        return None
    namespace, separator, local_name = element.tag[1:].partition("}")
    if not separator:
        return None
    return namespace, local_name.casefold()


def _element_is_link(element: Element) -> bool:
    name = _element_name(element)
    return name in {(XHTML_NAMESPACE, "a"), (SVG_NAMESPACE, "a")} and (
        "href" in element.attrib or XLINK_HREF_ATTRIBUTE in element.attrib
    )


def _element_hides_descendants(element: Element) -> bool:
    name = _element_name(element)
    if name is None:
        return False
    if "hidden" in element.attrib:
        return True
    return name == (XHTML_NAMESPACE, "dialog") and "open" not in element.attrib


def _picture_sources_before_image(
    children: Sequence[Element],
    *,
    image_index: int,
) -> tuple[tuple[tuple[Element, tuple[str, ...]], ...], bool]:
    """Return reachable sources and whether the image fallback URL is unreachable."""

    candidates: list[tuple[Element, tuple[str, ...]]] = []
    for element in children[:image_index]:
        if _element_name(element) != (XHTML_NAMESPACE, "source") or _element_hides_descendants(element):
            continue
        if not picture_source_is_potentially_eligible(element):
            continue
        sources = selectable_srcset_urls(element.attrib.get("srcset", ""))
        if not sources:
            continue
        candidates.append((element, sources))
        if picture_source_is_terminal(element, sources=sources):
            return tuple(candidates), True
    return tuple(candidates), False


def _occurrence_for_element(
    element: Element,
    *,
    marker_attribute: str,
    sources: Sequence[str],
    linked: bool,
) -> ImageOccurrence | None:
    marker = element.attrib.get(marker_attribute)
    if marker is None:
        return None
    line_text, separator, ordinal_text = marker.partition(":")
    if not separator or not line_text.isdigit() or not ordinal_text.isdigit():
        return None
    return ImageOccurrence(
        line_no=int(line_text),
        ordinal=int(ordinal_text),
        spec=ImageSpec(
            label=element.attrib.get("alt", ""),
            sources=tuple(source for source in sources if source),
            linked=linked,
        ),
    )


def _dom_image_occurrences(
    rendered_html: str,
    *,
    marker_attribute: str,
    raw_source_map: RawHTMLSourceMap,
) -> tuple[ImageOccurrence, ...]:
    parser = SourceMappedHTMLParser(
        marker_attribute=marker_attribute,
        raw_source_map=raw_source_map,
    )
    root = parser.parseFragment(rendered_html, container="div", scripting=True)
    occurrences: list[ImageOccurrence] = []
    seen_markers: set[str] = set()
    terminal_picture_images: set[int] = set()

    def record(element: Element, *, linked: bool, sources: Sequence[str]) -> None:
        marker = element.attrib.get(marker_attribute)
        if marker is None or marker in seen_markers:
            return
        occurrence = _occurrence_for_element(
            element,
            marker_attribute=marker_attribute,
            sources=sources,
            linked=linked,
        )
        if occurrence is not None:
            seen_markers.add(marker)
            occurrences.append(occurrence)

    stack: list[tuple[Element, bool]] = [(root, False)]
    while stack:
        element, ancestor_linked = stack.pop()
        name = _element_name(element)
        if name in NON_RENDERING_CONTAINERS or _element_hides_descendants(element):
            continue
        linked = ancestor_linked or _element_is_link(element)
        children = list(element)

        if name == (XHTML_NAMESPACE, "picture"):
            for image_index, child in enumerate(children):
                if _element_name(child) == (XHTML_NAMESPACE, "img") and not _element_hides_descendants(child):
                    previous_sources, is_terminal = _picture_sources_before_image(
                        children,
                        image_index=image_index,
                    )
                    for source_element, sources in previous_sources:
                        record(
                            source_element,
                            linked=linked,
                            sources=sources,
                        )
                    if is_terminal:
                        terminal_picture_images.add(id(child))
        elif name == (XHTML_NAMESPACE, "img"):
            record(
                element,
                linked=ancestor_linked,
                sources=() if id(element) in terminal_picture_images else html_image_sources(element),
            )
        elif name == (SVG_NAMESPACE, "image"):
            record(
                element,
                linked=ancestor_linked,
                sources=(element.attrib.get("href", ""), element.attrib.get(XLINK_HREF_ATTRIBUTE, "")),
            )

        stack.extend((child, linked) for child in reversed(children))
    return tuple(sorted(occurrences, key=lambda occurrence: occurrence.ordinal))


def rendered_image_occurrences(content: str) -> tuple[ImageOccurrence, ...]:
    """Return rendered image records for policy consumers."""

    rendered_html, marker_attribute, raw_source_map = render_markdown_with_markers(content)
    return _dom_image_occurrences(
        rendered_html,
        marker_attribute=marker_attribute,
        raw_source_map=raw_source_map,
    )


def rendered_markdown_images(content: str) -> tuple[RenderedMarkdownImage, ...]:
    """Return images that survive Markdown rendering and HTML visibility rules."""

    return tuple(
        RenderedMarkdownImage(
            line_no=occurrence.line_no,
            label=occurrence.spec.label,
            sources=occurrence.spec.sources,
            linked=occurrence.spec.linked,
        )
        for occurrence in rendered_image_occurrences(content)
    )
