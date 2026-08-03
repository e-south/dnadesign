"""Parse responsive-image candidates without choosing a browser viewport."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from xml.etree.ElementTree import Element

FLOATING_POINT_PATTERN = re.compile(r"-?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
NON_NEGATIVE_INTEGER_PATTERN = re.compile(r"[0-9]+")
ASCII_WHITESPACE = frozenset("\t\n\f\r ")
CSS_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
MIME_SUBTYPE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*")
MAX_IMAGE_DESCRIPTOR_INTEGER = "2147483647"


@dataclass(frozen=True, slots=True)
class _SrcsetCandidate:
    url: str
    width: str | None = None
    density: float | None = None
    future_height: str | None = None


def _positive_integer_descriptor(value: str) -> str | None:
    if NON_NEGATIVE_INTEGER_PATTERN.fullmatch(value) is None:
        return None
    normalized = value.lstrip("0")
    if not normalized:
        return None
    if len(normalized) > len(MAX_IMAGE_DESCRIPTOR_INTEGER) or (
        len(normalized) == len(MAX_IMAGE_DESCRIPTOR_INTEGER) and normalized > MAX_IMAGE_DESCRIPTOR_INTEGER
    ):
        return None
    return normalized


def _parse_descriptors(descriptors: Sequence[str]) -> tuple[str | None, float | None, str | None] | None:
    width: str | None = None
    density: float | None = None
    future_height: str | None = None
    for descriptor in descriptors:
        if descriptor.endswith("w") and (value := _positive_integer_descriptor(descriptor[:-1])) is not None:
            if width is not None or density is not None:
                return None
            width = value
            continue
        if descriptor.endswith("x") and FLOATING_POINT_PATTERN.fullmatch(descriptor[:-1]):
            value = float(descriptor[:-1])
            if (
                not math.isfinite(value)
                or value < 0
                or width is not None
                or density is not None
                or future_height is not None
            ):
                return None
            density = value
            continue
        if descriptor.endswith("h") and (value := _positive_integer_descriptor(descriptor[:-1])) is not None:
            if future_height is not None or density is not None:
                return None
            future_height = value
            continue
        return None
    if future_height is not None and width is None:
        return None
    return width, density, future_height


def _parse_srcset(value: str) -> tuple[_SrcsetCandidate, ...]:
    candidates: list[_SrcsetCandidate] = []
    position = 0
    while position < len(value):
        while position < len(value) and (value[position] in ASCII_WHITESPACE or value[position] == ","):
            position += 1
        if position >= len(value):
            break

        url_start = position
        while position < len(value) and value[position] not in ASCII_WHITESPACE:
            position += 1
        url = value[url_start:position]
        descriptors: list[str] = []
        if url.endswith(","):
            url = url.rstrip(",")
        else:
            while position < len(value) and value[position] in ASCII_WHITESPACE:
                position += 1
            descriptor: list[str] = []
            in_parentheses = False
            while position < len(value):
                character = value[position]
                if character == "," and not in_parentheses:
                    if descriptor:
                        descriptors.append("".join(descriptor))
                        descriptor = []
                    position += 1
                    break
                if character in ASCII_WHITESPACE and not in_parentheses:
                    if descriptor:
                        descriptors.append("".join(descriptor))
                        descriptor = []
                else:
                    if character == "(" and not in_parentheses:
                        in_parentheses = True
                    elif character == ")" and in_parentheses:
                        in_parentheses = False
                    descriptor.append(character)
                position += 1
            if descriptor:
                descriptors.append("".join(descriptor))

        parsed = _parse_descriptors(descriptors)
        if url and parsed is not None:
            width, density, future_height = parsed
            candidates.append(
                _SrcsetCandidate(
                    url=url,
                    width=width,
                    density=density,
                    future_height=future_height,
                )
            )
    return tuple(candidates)


def _selectable_srcset_candidates(value: str) -> tuple[_SrcsetCandidate, ...]:
    selected: list[_SrcsetCandidate] = []
    density_values: set[float] = set()
    width_values: set[str] = set()
    for candidate in _parse_srcset(value):
        if candidate.width is not None:
            if candidate.width in width_values:
                continue
            width_values.add(candidate.width)
        else:
            density = candidate.density if candidate.density is not None else 1.0
            if density in density_values:
                continue
            density_values.add(density)
        selected.append(candidate)
    if any(candidate.width is not None or candidate.density is None or candidate.density > 0 for candidate in selected):
        selected = [candidate for candidate in selected if candidate.density != 0]
    return tuple(selected)


def selectable_srcset_urls(value: str) -> tuple[str, ...]:
    """Return every candidate a conforming renderer could select."""

    return tuple(candidate.url for candidate in _selectable_srcset_candidates(value))


def html_image_sources(element: Element) -> tuple[str, ...]:
    """Return reachable ``img`` sources without assuming a viewport."""

    candidates = _selectable_srcset_candidates(element.attrib.get("srcset", ""))
    has_width = any(candidate.width is not None for candidate in candidates)
    has_density_one = any(
        candidate.width is None and (candidate.density is None or candidate.density == 1) for candidate in candidates
    )
    default_source = element.attrib.get("src", "")
    if default_source and not has_width and not has_density_one:
        candidates = tuple(candidate for candidate in candidates if candidate.density != 0)
    sources = [candidate.url for candidate in candidates]
    if default_source and not has_width and not has_density_one:
        sources.append(default_source)
    return tuple(sources)


def normalized_media_query(media: str) -> str:
    return " ".join(CSS_COMMENT_PATTERN.sub(" ", media).casefold().split())


def picture_source_is_potentially_eligible(element: Element) -> bool:
    media = element.attrib.get("media")
    if media is not None and normalized_media_query(media) == "not all":
        return False
    # Unknown queries remain inspectable because no viewport or CSS evaluator is available.
    declared_type = element.attrib.get("type")
    if declared_type is None:
        return True
    media_type = declared_type.partition(";")[0].strip().casefold()
    if not media_type:
        return True
    prefix, separator, subtype = media_type.partition("/")
    return separator == "/" and prefix == "image" and MIME_SUBTYPE_PATTERN.fullmatch(subtype) is not None


def picture_source_is_terminal(element: Element, *, sources: Sequence[str]) -> bool:
    """Return whether this source ends selection in every rendering environment."""

    media = element.attrib.get("media")
    media_is_unconditional = media is None or normalized_media_query(media) in {"", "all"}
    return bool(sources) and media_is_unconditional and "type" not in element.attrib


__all__ = [
    "html_image_sources",
    "picture_source_is_potentially_eligible",
    "picture_source_is_terminal",
    "selectable_srcset_urls",
]
