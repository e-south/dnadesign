"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_wt_mutation_scoring/selection.py

WT and position selection for Eco1 ESMC mutation scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.biohub_esmc_sae_profile.selection import (
    select_fold_accepted_biohub_esmc_sequences,
)

from .constants import WT_SEQUENCE_ID


@dataclass(frozen=True)
class Eco1WtMutationScoringSelection:
    """WT sequence and selected positions for masked-marginal scoring."""

    sequence_id: str
    sequence: str
    sequence_hash: str
    source_request_hash: str
    positions: tuple[int, ...]


def select_wt_mutation_scoring_sequence(
    *,
    output_root: Path,
    positions: str,
) -> Eco1WtMutationScoringSelection:
    """Select the accepted WT fold-check sequence and requested positions."""

    selection = select_fold_accepted_biohub_esmc_sequences(output_root=output_root, sequence_limit="1")
    record = selection.records[0]
    if record.sequence_id != WT_SEQUENCE_ID:
        raise ValueError(f"WT mutation scoring expected first selected sequence {WT_SEQUENCE_ID!r}")
    selected_positions = parse_position_selection(positions, sequence_length=len(record.sequence))
    return Eco1WtMutationScoringSelection(
        sequence_id=record.sequence_id,
        sequence=record.sequence,
        sequence_hash=record.sequence_hash,
        source_request_hash=selection.source_request_hash,
        positions=selected_positions,
    )


def parse_position_selection(raw: str, *, sequence_length: int) -> tuple[int, ...]:
    """Parse comma-separated one-based positions and inclusive ranges."""

    text = str(raw or "").strip().lower()
    if text == "all":
        return tuple(range(1, sequence_length + 1))
    positions: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"position range start exceeds end: {token}")
            positions.extend(range(start, end + 1))
        else:
            positions.append(int(token))
    if not positions:
        raise ValueError("positions must be 'all' or a comma-separated list of one-based positions/ranges")
    bad = [position for position in positions if position < 1 or position > sequence_length]
    if bad:
        raise ValueError(f"position(s) out of bounds for WT length {sequence_length}: {bad}")
    if len(set(positions)) != len(positions):
        raise ValueError(f"duplicate positions are not allowed: {positions}")
    return tuple(positions)
