"""
Materialize mutable sequence-view semantics for the promoter study.

The stable sequence-view sidecar answers "what sequence product is this?".
This helper writes the companion addendum that answers study/provenance
questions such as source family, selection basis, and view collection without
changing `view_id`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.usr import (
    Dataset,
    ViewSemanticsRecord,
    load_sequence_views,
    load_view_semantics_index,
    write_view_semantics,
)
from dnadesign.usr.src.storage.parquet import now_utc

DEFAULT_DATASETS = (
    "usr_promoter_references",
    "construct_prom_eth_cip_reference_core60",
    "usr_prom_eth_cip_anchor",
    "construct_prom_eth_cip_context",
    "construct_prom_eth_cip_reference_contexts",
)
DEFAULT_STUDY_ID = "stress_ethanol_cipro_growth"


@dataclass(frozen=True)
class PromoterStudyViewSemanticsResult:
    datasets: list[str]
    views_seen: int
    semantics_planned: int
    semantics_written: int
    existing_semantics: int
    by_source_family: dict[str, int]
    by_selection_basis: dict[str, int]
    by_view_collection: dict[str, int]


def _default_usr_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets"


def _record_source_by_id(dataset: Dataset) -> dict[str, str | None]:
    path = dataset.records_path
    if not path.exists():
        return {}
    table = pq.read_table(path, columns=["id", "source"])
    out: dict[str, str | None] = {}
    for row in table.to_pylist():
        row_id = row.get("id")
        if row_id is None:
            continue
        source = row.get("source")
        out[str(row_id)] = str(source).strip() if source is not None and str(source).strip() else None
    return out


def _record_source_family(source: str | None) -> str:
    text = (source or "").strip().casefold()
    if not text:
        return "unknown_source"
    if "reader_sfxi" in text or "sfxi" in text:
        return "sfxi_archive"
    if text.startswith("plan_pool__") or "densegen" in text:
        return "densegen_generated"
    if "construct_prom_eth_cip_reference_core60" in text:
        return "construct_derived"
    if "projected_promoter_reference_insert" in text:
        return "reference_source"
    if "mg1655" in text:
        return "legacy_reference_control"
    if "construct seed" in text:
        return "legacy_construct_seed"
    return "curated_source"


def _semantics_for_view(
    *,
    dataset_name: str,
    view: Any,
    record_sources: dict[str, str | None],
    created_at: str,
    study_id: str,
    created_by: str,
) -> ViewSemanticsRecord:
    product_kind = str(view.product_kind)
    orientation = str(view.orientation)
    source_family: str
    selection_basis: str
    view_collections: list[str]
    role_tags: list[str]

    if product_kind == "selected_region":
        derivation_spec = str(view.derivation_spec_id or "")
        if derivation_spec.startswith("synthetic_promoter_standard:"):
            source_family = "synthetic_reference_standard"
            selection_basis = "curated_synthetic_standard"
            view_collections = ["reference_source_inserts", "synthetic_reference_standards"]
        elif derivation_spec.startswith("project_genbank_upstream:"):
            source_family = "genbank_projected_reference"
            selection_basis = "genbank_projected_upstream_region"
            view_collections = ["reference_source_inserts", "genbank_projected_references"]
        else:
            source_family = "reference_source"
            selection_basis = "curated_reference_insert"
            view_collections = ["reference_source_inserts"]
        role_tags = ["reference_control"]
    elif product_kind == "analysis_window":
        source_family = "construct_derived"
        selection_basis = "sigma_site_pair_midpoint"
        view_collections = ["reference_analysis_window_comparison"]
        role_tags = ["comparability_view", "reference_control"]
    elif product_kind == "construct_insert":
        record_source = record_sources.get(str(view.sequence_id))
        source_family = _record_source_family(record_source)
        if bool(view.analysis_only) or view.parent_dataset_id == "construct_prom_eth_cip_reference_core60":
            source_family = "construct_derived"
            selection_basis = "sigma_site_pair_midpoint"
            view_collections = ["merged_anchor_handoff", "reference_analysis_window_comparison"]
            role_tags = ["construct_ready_insert", "comparability_view"]
        elif source_family == "sfxi_archive":
            selection_basis = "archive_backed_insert"
            view_collections = ["merged_anchor_handoff", "sfxi_archive_handoff"]
            role_tags = ["construct_ready_insert", "design_population", "archive_source"]
        elif source_family in {"reference_source", "legacy_reference_control"}:
            selection_basis = "native_source_length"
            view_collections = ["merged_anchor_handoff", "reference_source_inserts"]
            role_tags = ["construct_ready_insert", "reference_control"]
        elif source_family == "legacy_construct_seed":
            selection_basis = "legacy_construct_seed"
            view_collections = ["merged_anchor_handoff", "legacy_control_handoff"]
            role_tags = ["construct_ready_insert", "legacy_control"]
        else:
            selection_basis = "densegen_selected_insert"
            view_collections = ["merged_anchor_handoff"]
            role_tags = ["construct_ready_insert", "design_population"]
    elif product_kind == "realized_context":
        source_family = "construct_derived"
        if orientation == "reverse_complement":
            selection_basis = "whole_output_reverse_complement"
            orientation_collection = "realized_context_reverse_complement_all"
            role_tags = ["context_exposure", "orientation_sensitivity"]
        else:
            selection_basis = "template_window_center"
            orientation_collection = "realized_context_forward_all"
            role_tags = ["context_exposure"]
        if dataset_name == "construct_prom_eth_cip_reference_contexts":
            view_collections = [orientation_collection, "reference_realized_context_comparison"]
            role_tags = [*role_tags, "reference_control"]
        else:
            view_collections = [orientation_collection, "merged_context_handoff"]
    else:
        raise ValueError(f"Unsupported product_kind '{product_kind}' in dataset '{dataset_name}'.")

    return ViewSemanticsRecord(
        view_id=str(view.view_id),
        sequence_id=str(view.sequence_id),
        source_family=source_family,
        selection_basis=selection_basis,
        view_collections=view_collections,
        role_tags=role_tags,
        study_id=study_id,
        created_at=created_at,
        created_by=created_by,
    )


def _matches_existing(existing: dict[str, object] | None, row: ViewSemanticsRecord) -> bool:
    if existing is None:
        return False
    expected = row.model_dump(mode="python")
    expected.pop("created_at", None)
    expected.pop("created_by", None)
    observed = dict(existing)
    observed.pop("created_at", None)
    observed.pop("created_by", None)
    return observed == expected


def materialize_promoter_study_view_semantics(
    *,
    usr_root: Path,
    dataset_names: list[str] | None = None,
    study_id: str = DEFAULT_STUDY_ID,
    write: bool = False,
) -> PromoterStudyViewSemanticsResult:
    names = list(dataset_names or DEFAULT_DATASETS)
    unsupported = [name for name in names if name not in DEFAULT_DATASETS]
    if unsupported:
        preview = ", ".join(unsupported)
        raise ValueError(f"Unsupported promoter-study semantics dataset(s): {preview}")

    created_at = now_utc()
    created_by = "dnadesign.usr.materialize_promoter_study_view_semantics"
    planned_rows: list[tuple[Dataset, ViewSemanticsRecord]] = []
    existing_total = 0
    source_counts: Counter[str] = Counter()
    selection_counts: Counter[str] = Counter()
    collection_counts: Counter[str] = Counter()
    views_seen = 0

    for dataset_name in names:
        dataset = Dataset(usr_root, dataset_name)
        dataset._require_exists()  # noqa: SLF001
        views = load_sequence_views(dataset)
        views_seen += len(views)
        record_sources = _record_source_by_id(dataset)
        existing = load_view_semantics_index(dataset)
        existing_total += len(existing)
        for view in views:
            row = _semantics_for_view(
                dataset_name=dataset.name,
                view=view,
                record_sources=record_sources,
                created_at=created_at,
                study_id=study_id,
                created_by=created_by,
            )
            source_counts[str(row.source_family)] += 1
            selection_counts[str(row.selection_basis)] += 1
            for collection in row.view_collections or []:
                collection_counts[str(collection)] += 1
            current = existing.get(str(row.view_id))
            if current is None:
                planned_rows.append((dataset, row))
                continue
            if not _matches_existing(current, row):
                raise ValueError(f"Existing view-semantics row '{row.view_id}' has non-idempotent metadata drift.")

    written = 0
    if write:
        rows_by_dataset: dict[str, tuple[Dataset, list[ViewSemanticsRecord]]] = {}
        for dataset, row in planned_rows:
            _, rows = rows_by_dataset.setdefault(dataset.name, (dataset, []))
            rows.append(row)
        for dataset, rows in rows_by_dataset.values():
            written += write_view_semantics(
                dataset,
                rows,
                conflict_policy="idempotent",
                actor={
                    "tool": "usr",
                    "run_id": "materialize_promoter_study_view_semantics",
                    "command": "materialize_promoter_study_view_semantics",
                    "dataset": dataset.name,
                },
            )

    return PromoterStudyViewSemanticsResult(
        datasets=names,
        views_seen=views_seen,
        semantics_planned=len(planned_rows),
        semantics_written=written,
        existing_semantics=existing_total,
        by_source_family=dict(sorted(source_counts.items())),
        by_selection_basis=dict(sorted(selection_counts.items())),
        by_view_collection=dict(sorted(collection_counts.items())),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize view_semantics addenda for promoter-study sequence views."
    )
    parser.add_argument("--usr-root", type=Path, default=_default_usr_root())
    parser.add_argument("--dataset", action="append", dest="datasets", help="Dataset to materialize. Repeatable.")
    parser.add_argument("--study-id", default=DEFAULT_STUDY_ID)
    parser.add_argument("--write", action="store_true", help="Write missing addendum rows. Default is dry-run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize_promoter_study_view_semantics(
        usr_root=args.usr_root,
        dataset_names=args.datasets,
        study_id=args.study_id,
        write=bool(args.write),
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
