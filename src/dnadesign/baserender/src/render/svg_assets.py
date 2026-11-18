"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/svg_assets.py

Select pinned SVG geometry and its definition closure without rasterization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import hashlib
import re
from xml.etree import ElementTree as ET

from ..core.errors import RenderingError
from ..core.vector import SvgArtwork

SVG_NS = "http://www.w3.org/2000/svg"
SVG = f"{{{SVG_NS}}}"
_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)
_BLOCKED = {"image", "script", "foreignObject", "style", "animate", "animateMotion", "animateTransform", "set"}
_MAX_BYTES = 64 * 1024 * 1024


def _references(node):
    if not isinstance(node.tag, str) or not node.tag.startswith(SVG) or node.tag[len(SVG) :] in _BLOCKED:
        raise RenderingError(f"unsupported element in vector selection: {node.tag!r}")
    refs = set()
    for key, value in node.attrib.items():
        local = key.split("}")[-1]
        if (
            local.lower().startswith("on")
            or "@import" in value.lower()
            or "\\" in value
            or "/*" in value
            or local == "base"
        ):
            raise RenderingError("active content is not supported in SVG artwork")
        if local == "href":
            if not value.startswith("#") or len(value) == 1:
                raise RenderingError("SVG artwork references must be local fragment IDs")
            refs.add(value[1:])
        for match in _URL.finditer(value):
            target = match.group(2).strip()
            if not target.startswith("#") or len(target) == 1:
                raise RenderingError("SVG artwork url() references must be local fragment IDs")
            refs.add(target[1:])
    return refs


def load_svg_selection(artwork: SvgArtwork):
    artwork.validate()
    try:
        with artwork.path.open("rb") as handle:
            raw = handle.read(_MAX_BYTES + 1)
    except OSError as exc:
        raise RenderingError(f"cannot read SVG artwork: {artwork.path.name}") from exc
    if len(raw) > _MAX_BYTES:
        raise RenderingError("SVG artwork exceeds the 64-MiB source limit; select a smaller source")
    if hashlib.sha256(raw).hexdigest() != artwork.sha256:
        raise RenderingError("SVG artwork SHA-256 mismatch; review and explicitly select the new source")
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise RenderingError("SVG artwork must not contain DTDs or entity declarations")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise RenderingError("malformed SVG artwork") from exc
    if root.tag != f"{SVG}svg":
        raise RenderingError("SVG artwork must have an SVG-namespace svg root")
    if root.find(f".//{SVG}style") is not None:
        raise RenderingError("SVG stylesheets are not supported in selections; use inline presentation attributes")
    by_id = {}
    parents = {}
    for node in root.iter():
        for child in node:
            parents[child] = node
        if key := node.get("id"):
            if key in by_id:
                raise RenderingError(f"duplicate SVG source ID: {key!r}")
            by_id[key] = node
    missing = sorted(set(artwork.element_ids) - by_id.keys())
    if missing:
        raise RenderingError(f"missing SVG selections: {missing}")
    selected = {by_id[key] for key in artwork.element_ids}
    for node in selected:
        parent = parents.get(node)
        while parent is not None:
            if parent in selected:
                raise RenderingError("SVG selections overlap; select each parent or its children, not both")
            parent = parents.get(parent)

    needed = set()
    visiting = set()

    def visit(node, depth=0):
        if depth > 128:
            raise RenderingError("SVG definition or element nesting exceeds 128 levels")
        if node in visiting:
            raise RenderingError("cyclic SVG definition reference")
        if node in needed:
            return
        visiting.add(node)
        for key in _references(node):
            if key not in by_id:
                raise RenderingError(f"missing SVG definition: {key!r}")
            visit(by_id[key], depth + 1)
        for child in node:
            visit(child, depth + 1)
        visiting.remove(node)
        needed.add(node)

    for node in selected:
        visit(node)
        parent = parents.get(node)
        while parent is not None:
            for key in _references(parent):
                if key not in by_id:
                    raise RenderingError(f"missing SVG ancestor definition: {key!r}")
                visit(by_id[key])
            parent = parents.get(parent)

    def prune(node, depth=0):
        if depth > 128:
            raise RenderingError("SVG source nesting exceeds 128 levels")
        children = [result for child in node if (result := prune(child, depth + 1)) is not None]
        if node not in needed and node is not root and not children:
            return None
        result = ET.Element(node.tag, dict(node.attrib))
        result.text = node.text if node in needed else None
        result.tail = None
        result.extend(children)
        return result

    fragment = prune(root)
    # Root dimensions come from the declared selection frame, not source-page metadata.
    fragment.attrib = {
        key: value
        for key, value in fragment.attrib.items()
        if key not in {"x", "y", "width", "height", "viewBox", "id", "version"}
        and (not key.startswith("{") or key.startswith("{http://www.w3.org/XML/1998/namespace}"))
    }
    return fragment


def namespace_selection(fragment, prefix):
    result = copy.deepcopy(fragment)
    ids = {node.get("id"): f"{prefix}-{node.get('id')}" for node in result.iter() if node.get("id")}
    for node in result.iter():
        for key, value in list(node.attrib.items()):
            if key == "id":
                node.set(key, ids[value])
            elif key.split("}")[-1] == "href" and value.startswith("#"):
                node.set(key, f"#{ids[value[1:]]}")
            elif "url(" in value.lower():
                node.set(key, _URL.sub(lambda match: f"url(#{ids[match.group(2).strip()[1:]]})", value))
    return result
