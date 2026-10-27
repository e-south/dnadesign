"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/rnafold.py

ViennaRNA RNAfold-format parsing helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from dnadesign.contracts.folding.secondary_structure_prediction_v2 import (
    SecondaryStructurePairV1,
    SecondaryStructurePredictionResultV1,
)

from .errors import FoldingLengthMismatchError, FoldingMalformedOutputError

_DOT_BRACKET_RE = re.compile(r"^\s*([.()\[\]{}<>]+)\s*(?:\(\s*([-+]?\d+(?:\.\d+)?)\s*\))?")
_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}", "<": ">"}
_CLOSE_TO_OPEN = {close: open_ for open_, close in _OPEN_TO_CLOSE.items()}


def parse_rnafold_stdout(
    *,
    stdout: str,
    submitted_sequence: str,
    input_length: int,
) -> SecondaryStructurePredictionResultV1:
    dot_bracket, mfe = _extract_dot_bracket(stdout)
    if len(dot_bracket) != input_length:
        raise FoldingLengthMismatchError(
            f"ViennaRNA dot-bracket length {len(dot_bracket)} does not match input length {input_length}."
        )
    pair_map = _pair_map_from_dot_bracket(dot_bracket, submitted_sequence=submitted_sequence)
    return SecondaryStructurePredictionResultV1(
        dot_bracket=dot_bracket,
        mfe_kcal_mol=mfe,
        pair_map=pair_map,
    )


def _extract_dot_bracket(stdout: str) -> tuple[str, float | None]:
    for line in stdout.splitlines():
        if line.lstrip().startswith(">"):
            continue
        match = _DOT_BRACKET_RE.match(line)
        if match is None:
            continue
        dot_bracket = match.group(1)
        mfe_text = match.group(2)
        return dot_bracket, float(mfe_text) if mfe_text is not None else None
    raise FoldingMalformedOutputError("ViennaRNA output did not contain a dot-bracket structure line.")


def _pair_map_from_dot_bracket(
    dot_bracket: str,
    *,
    submitted_sequence: str,
) -> list[SecondaryStructurePairV1]:
    if len(submitted_sequence) != len(dot_bracket):
        raise FoldingLengthMismatchError("Submitted sequence length does not match dot-bracket length.")
    stacks: dict[str, list[int]] = {opener: [] for opener in _OPEN_TO_CLOSE}
    pairs: list[SecondaryStructurePairV1] = []
    for index, char in enumerate(dot_bracket):
        if char == ".":
            continue
        if char in _OPEN_TO_CLOSE:
            stacks[char].append(index)
            continue
        opener = _CLOSE_TO_OPEN.get(char)
        if opener is None:
            raise FoldingMalformedOutputError(f"Unsupported dot-bracket character '{char}'.")
        if not stacks[opener]:
            raise FoldingMalformedOutputError("ViennaRNA dot-bracket has invalid bracket nesting.")
        left = stacks[opener].pop()
        pair = submitted_sequence[left].upper() + submitted_sequence[index].upper()
        pairs.append(SecondaryStructurePairV1(left=left, right=index, pair=pair))
    if any(stack for stack in stacks.values()):
        raise FoldingMalformedOutputError("ViennaRNA dot-bracket has invalid bracket nesting.")
    return sorted(pairs, key=lambda item: (item.left, item.right))


__all__ = ["parse_rnafold_stdout"]
