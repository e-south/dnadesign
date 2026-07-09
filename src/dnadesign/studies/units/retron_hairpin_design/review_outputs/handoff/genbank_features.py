"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/handoff/genbank_features.py

GenBank feature-direction normalization for Retron MSD handoff records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

from ...compiler.exceptions import RetronMsdCompilerError
from ..contracts.feature_directions import feature_direction_for_role


def rewrite_reverse_complement_features(lines: Sequence[str]) -> list[str]:
    feature_blocks = _split_feature_blocks(lines)
    has_cap = any(_feature_role(block) == "snapback_cap" for block in feature_blocks)
    rewritten = [lines[0]]
    for block in feature_blocks:
        role = _feature_role(block)
        normalized = _normalize_feature_block(block, role=role)
        rewritten.extend(normalized)
        if role == "snapback_foldback_geometry" and not has_cap:
            rewritten.extend(_cap_block_from_foldback(normalized))
    return rewritten


def _split_feature_blocks(lines: Sequence[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines[1:]:
        if line.startswith("     ") and not line.startswith("                     "):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _normalize_feature_block(block: Sequence[str], *, role: str) -> list[str]:
    direction = feature_direction_for_role(role)
    if direction is not None:
        return _set_feature_direction(block, direction=direction)
    if role:
        raise RetronMsdCompilerError(f"Retron Benchling GenBank feature has unknown dnadesign_role: {role}")
    return list(block)


def _set_feature_direction(block: Sequence[str], *, direction: str) -> list[str]:
    lines = list(block)
    location = _plain_location(lines[0])
    if direction == "reverse":
        lines[0] = f"{lines[0][:21]}complement({location})"
        return _set_strand(lines, "-1")
    lines[0] = f"{lines[0][:21]}{location}"
    if direction == "forward":
        return _set_strand(lines, "1")
    return _set_strand(lines, None)


def _plain_location(feature_line: str) -> str:
    location = feature_line[21:].strip()
    if location.startswith("complement(") and location.endswith(")"):
        return location.removeprefix("complement(").removesuffix(")")
    return location


def _set_strand(lines: Sequence[str], strand: str | None) -> list[str]:
    without_strand = [line for line in lines if "/strand=" not in line]
    if strand is None:
        return without_strand
    return [*without_strand, f'                     /strand="{strand}"']


def _cap_block_from_foldback(block: Sequence[str]) -> list[str]:
    replacements = {
        '/label="Foldback"': '/label="Cap"',
        '/dnadesign_feature_id="snapback_foldback_geometry"': '/dnadesign_feature_id="snapback_cap"',
        '/dnadesign_role="snapback_foldback_geometry"': '/dnadesign_role="snapback_cap"',
    }
    return [_replace_qualifier(line, replacements) for line in block]


def _replace_qualifier(line: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        if old in line:
            return line.replace(old, new)
    return line


def _feature_role(block: Sequence[str]) -> str:
    for line in block:
        marker = '/dnadesign_role="'
        if marker in line:
            return line.split(marker, maxsplit=1)[1].split('"', maxsplit=1)[0]
    return ""


__all__ = ["rewrite_reverse_complement_features"]
