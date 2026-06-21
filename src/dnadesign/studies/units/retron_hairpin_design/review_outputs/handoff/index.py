"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/handoff/index.py

Review-local sequence handoff index generation for Retron review packages.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..sequence.index import SequenceReviewFrame
from ..sequence.variant_identity import identity_for_frame
from .contract import SEQUENCE_HANDOFF_COLUMNS


@dataclass(frozen=True)
class HandoffIndex:
    tsv_path: Path
    markdown_path: Path


def write_handoff_index(
    frames: Sequence[SequenceReviewFrame],
    *,
    review_root: Path,
    materialized_root: Path,
    deliverable_plan_id: str,
) -> HandoffIndex:
    handoff_dir = review_root / "reviews" / "handoff"
    tsv_path = handoff_dir / f"{deliverable_plan_id}.handoff.tsv"
    markdown_path = handoff_dir / f"{deliverable_plan_id.replace('_', '-')}.handoff.md"
    rows = [_handoff_row(frame=frame, review_root=review_root, materialized_root=materialized_root) for frame in frames]
    handoff_dir.mkdir(parents=True, exist_ok=True)
    _write_tsv(tsv_path, rows)
    _write_markdown(markdown_path, rows)
    return HandoffIndex(tsv_path=tsv_path, markdown_path=markdown_path)


def _handoff_row(*, frame: SequenceReviewFrame, review_root: Path, materialized_root: Path) -> dict[str, str]:
    row = frame.row
    identity = identity_for_frame(frame)
    return {
        "order": str(frame.order),
        "variant_id": identity.variant_id,
        "construct_id": frame.construct_id,
        "msd_design_id": frame.msd_design_id,
        "scaffold": identity.scaffold,
        "retained_window": identity.retained_window,
        "insert_nt": str(identity.insert_nt),
        "role": identity.role,
        "genbank": _review_relative(materialized_root / row["genbank"], review_root=review_root),
        "reverse_complement_genbank": _review_relative(
            materialized_root / row["reverse_complement_genbank"],
            review_root=review_root,
        ),
        "forward_fasta": _review_relative(materialized_root / row["forward_fasta"], review_root=review_root),
        "reverse_complement_fasta": _review_relative(
            materialized_root / row["reverse_complement_fasta"],
            review_root=review_root,
        ),
        "features_csv": _review_relative(materialized_root / row["features_csv"], review_root=review_root),
    }


def _write_tsv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SEQUENCE_HANDOFF_COLUMNS), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: Sequence[dict[str, str]]) -> None:
    lines = [
        "# tetO Trim Sequence Handoff",
        "",
        "One row per compiled variant. Open GenBank first; FASTA and feature CSV sidecars are linked next to it. "
        "Full machine metadata stays in `sequence_index.tsv` and `review_manifest.json`.",
        "",
        "| Variant | Insert | Context | Files |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(row["variant_id"]),
                    _escape_md(f"{row['insert_nt']} nt {row['retained_window']}"),
                    _escape_md(f"{row['scaffold']} {row['role']}"),
                    " ".join(
                        [
                            _md_link("GB", row["genbank"], markdown_path=path),
                            _md_link("RC GB", row["reverse_complement_genbank"], markdown_path=path),
                            _md_link("FA", row["forward_fasta"], markdown_path=path),
                            _md_link("RC FA", row["reverse_complement_fasta"], markdown_path=path),
                            _md_link("CSV", row["features_csv"], markdown_path=path),
                        ]
                    ),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md_link(label: str, review_relative_path: str, *, markdown_path: Path) -> str:
    target = Path("..") / ".." / review_relative_path
    return f"[{label}]({target.as_posix()})"


def _review_relative(path: Path, *, review_root: Path) -> str:
    try:
        return path.resolve().relative_to(review_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|")


__all__ = ["HandoffIndex", "write_handoff_index"]
