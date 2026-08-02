"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/outputs/names.py

Builds deterministic filenames for BaseRender output writers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib

_MAX_STEM_BYTES = 120
_STEM_DIGEST_LENGTH = 16
_HASH_CHUNK_CHARS = 4_096
_ALLOWED_STEM_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
_EDGE_STEM_CHARS = frozenset("._-")


def _bounded_sanitized_stem(raw: str) -> tuple[str, bool]:
    output: list[str] = []
    pending_edge: list[str] = []
    pending_edge_count = 0
    in_invalid_run = False
    overflow = False

    def append(character: str) -> None:
        nonlocal overflow
        if len(output) < _MAX_STEM_BYTES + 1:
            output.append(character)
        else:
            overflow = True

    for character in raw:
        if character in _ALLOWED_STEM_CHARS:
            token = character
            in_invalid_run = False
        elif in_invalid_run:
            continue
        else:
            token = "_"
            in_invalid_run = True
        if token in _EDGE_STEM_CHARS:
            if output:
                pending_edge_count += 1
                if len(pending_edge) < _MAX_STEM_BYTES + 1:
                    pending_edge.append(token)
            continue
        for pending in pending_edge:
            append(pending)
        if pending_edge_count > len(pending_edge):
            overflow = True
        pending_edge.clear()
        pending_edge_count = 0
        append(token)
    return "".join(output), overflow


def _raw_digest(raw: str) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(raw), _HASH_CHUNK_CHARS):
        digest.update(raw[start : start + _HASH_CHUNK_CHARS].encode("utf-8", errors="surrogatepass"))
    return digest.hexdigest()[:_STEM_DIGEST_LENGTH]


def _safe_stem(raw: str) -> str:
    stem, overflow = _bounded_sanitized_stem(raw)
    stem = stem or "record"
    if not overflow and len(stem) <= _MAX_STEM_BYTES:
        return stem
    digest = _raw_digest(raw)
    prefix_bytes = _MAX_STEM_BYTES - _STEM_DIGEST_LENGTH - 1
    prefix = stem[:prefix_bytes]
    return f"{prefix}_{digest}"


def _unique_stem(base: str, used: set[str]) -> str:
    normalized = base.casefold()
    if normalized not in used:
        used.add(normalized)
        return base
    i = 2
    while True:
        candidate = f"{base}_{i}"
        normalized_candidate = candidate.casefold()
        if normalized_candidate not in used:
            used.add(normalized_candidate)
            return candidate
        i += 1


__all__ = ["_safe_stem", "_unique_stem"]
