"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/handoff/benchling.py

Benchling GenBank import folder generation for Retron review packages.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ...compiler.exceptions import RetronMsdCompilerError
from ..contracts.benchling_import import BENCHLING_GENBANK_DIRNAME, BenchlingGenbankImportPlan
from ..sequence.index import SequenceReviewFrame
from ..sequence.variant_identity import identity_for_frame

BENCHLING_INDEX_COLUMNS = (
    "order",
    "variant_id",
    "assigned_construct_id",
    "filename",
    "description",
    "source_construct_id",
    "source_reverse_complement_genbank",
)


@dataclass(frozen=True)
class BenchlingGenbankExport:
    directory: Path
    index_tsv: Path
    files: tuple[Path, ...]


def write_benchling_genbank_import(
    frames: Sequence[SequenceReviewFrame],
    *,
    review_root: Path,
    materialized_root: Path,
    deliverable_plan_id: str,
    benchling_plan: BenchlingGenbankImportPlan,
) -> BenchlingGenbankExport:
    target_dir = review_root / BENCHLING_GENBANK_DIRNAME
    index_path = review_root / "reviews" / "handoff" / f"{deliverable_plan_id}.benchling_genbank.tsv"
    selected = [frame for frame in frames if frame.payload_trim_id in benchling_plan.included_payload_trim_ids]
    selected_ids = [identity_for_frame(frame).variant_id for frame in selected]
    expected_ids = list(benchling_plan.expected_variant_ids)
    if selected_ids != expected_ids:
        raise RetronMsdCompilerError(
            "Retron Benchling GenBank export expected trim frames in design-set order "
            f"{expected_ids}, observed {selected_ids}."
        )
    if len(selected) != benchling_plan.expected_count:
        raise RetronMsdCompilerError(
            "Retron Benchling GenBank export selected count does not match deliverable plan: "
            f"{len(selected)} != {benchling_plan.expected_count}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    _clear_existing_genbanks(target_dir)
    rows: list[dict[str, str]] = []
    written: list[Path] = []
    for frame in selected:
        identity = identity_for_frame(frame)
        assigned_construct_id = benchling_plan.assigned_construct_id(identity.variant_id)
        filename = benchling_plan.filename_for(identity.variant_id)
        description = benchling_plan.description_for(identity.variant_id)
        source = materialized_root / frame.row["reverse_complement_genbank"]
        target = target_dir / filename
        _write_relabelled_genbank(
            source=source,
            target=target,
            assigned_construct_id=assigned_construct_id,
            payload_label=benchling_plan.filename_payload_label,
            description=description,
            source_construct_id=frame.construct_id,
            sequence_length=int(frame.row["sequence_length"]),
        )
        written.append(target)
        rows.append(
            {
                "order": str(frame.order),
                "variant_id": identity.variant_id,
                "assigned_construct_id": assigned_construct_id,
                "filename": filename,
                "description": description,
                "source_construct_id": frame.construct_id,
                "source_reverse_complement_genbank": _review_relative(source, review_root=review_root),
            }
        )
    _write_index(index_path, rows)
    _assert_import_folder_is_flat(target_dir, expected_files={path.name for path in written})
    return BenchlingGenbankExport(directory=target_dir, index_tsv=index_path, files=tuple(written))


def _write_relabelled_genbank(
    *,
    source: Path,
    target: Path,
    assigned_construct_id: str,
    payload_label: str,
    description: str,
    source_construct_id: str,
    sequence_length: int,
) -> None:
    if not source.is_file():
        raise RetronMsdCompilerError(f"Retron Benchling GenBank source not found: {source}")
    lines = source.read_text(encoding="utf-8").splitlines()
    try:
        features_index = next(index for index, line in enumerate(lines) if line.startswith("FEATURES"))
    except StopIteration as exc:
        raise RetronMsdCompilerError(f"Retron Benchling GenBank source has no FEATURES section: {source}") from exc
    definition = (
        f"{assigned_construct_id}-msd[{payload_label}]; {description}; "
        f"reverse-complement MSD handoff from {source_construct_id}."
    )
    header = [
        f"LOCUS       {assigned_construct_id:<16}{sequence_length:>11} bp ss-DNA linear SYN",
        f"DEFINITION  {definition}",
        f"ACCESSION   {assigned_construct_id}",
    ]
    target.write_text("\n".join([*header, *lines[features_index:]]) + "\n", encoding="utf-8")


def _clear_existing_genbanks(target_dir: Path) -> None:
    for path in target_dir.iterdir():
        if path.name.startswith("."):
            continue
        if path.is_file() and path.suffix == ".gb":
            path.unlink()


def _assert_import_folder_is_flat(target_dir: Path, *, expected_files: set[str]) -> None:
    observed = {path.name for path in target_dir.iterdir() if path.is_file() and not path.name.startswith(".")}
    if observed != expected_files:
        raise RetronMsdCompilerError(
            "Retron Benchling GenBank import folder must contain only the expected .gb files: "
            f"{sorted(observed)} != {sorted(expected_files)}"
        )


def _write_index(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BENCHLING_INDEX_COLUMNS), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _review_relative(path: Path, *, review_root: Path) -> str:
    try:
        return path.resolve().relative_to(review_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


__all__ = [
    "BENCHLING_GENBANK_DIRNAME",
    "BENCHLING_INDEX_COLUMNS",
    "BenchlingGenbankExport",
    "write_benchling_genbank_import",
]
