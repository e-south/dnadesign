"""
Materialize merged promoter-anchor sequence views without inventing analysis windows.

The merged study anchor handoff is a construct-ready collection of promoter
inserts. Some rows are natively 60 bp; some are native biological inserts of
other lengths; some are derived analysis-core rows. This helper writes one
`construct_insert` sequence view per base row and preserves analysis-window
lineage as metadata rather than duplicating native 60 bp rows as
`analysis_window`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.usr import (
    Dataset,
    SequenceViewRecord,
    load_sequence_view_index,
    write_sequence_views,
)
from dnadesign.usr.src.storage.parquet import now_utc

DEFAULT_DATASET = "usr_prom_eth_cip_anchor"


@dataclass(frozen=True)
class AnchorSequenceViewResult:
    dataset: str
    rows_seen: int
    views_planned: int
    views_written: int
    existing_views: int
    analysis_only_views: int
    biological_derived_rows: int
    analysis_window_source_rows: int


def _default_usr_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets"


def _read_optional_overlay(dataset: Dataset, namespace: str) -> list[dict[str, Any]]:
    path = dataset.dir / "_derived" / f"{namespace}.parquet"
    if not path.exists():
        return []
    return [dict(row) for row in pq.read_table(path).to_pylist()]


def _clean_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _view_name(*, row_id: str, label: str | None) -> str:
    if label:
        safe = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in label)
        return f"{safe}_construct_insert"
    return f"{row_id[:12]}_construct_insert"


def _source_interval_fields(derived: dict[str, Any]) -> dict[str, int | None]:
    parent_id = _clean_text(derived.get("derived__parent_id"))
    parent_dataset = _clean_text(derived.get("derived__parent_dataset"))
    if parent_id is None or parent_dataset is None:
        return {"source_interval_start_0": None, "source_interval_end_0": None}
    return {
        "source_interval_start_0": derived.get("derived__source_interval_start_0"),
        "source_interval_end_0": derived.get("derived__source_interval_end_0"),
    }


def _build_anchor_sequence_views(dataset: Dataset) -> list[SequenceViewRecord]:
    records = pq.read_table(dataset.records_path, columns=["id"]).to_pylist()
    labels = {
        str(row["id"]): _clean_text(row.get("usr_label__primary"))
        for row in _read_optional_overlay(dataset, "usr_label")
        if row.get("id") is not None
    }
    derived_by_id = {
        str(row["id"]): row for row in _read_optional_overlay(dataset, "derived") if row.get("id") is not None
    }
    created_at = now_utc()
    views: list[SequenceViewRecord] = []
    for row in records:
        row_id = str(row["id"])
        derived = derived_by_id.get(row_id, {})
        derived_product = _clean_text(derived.get("derived__product_kind"))
        analysis_only = bool(derived.get("derived__analysis_only")) or derived_product == "analysis_window"
        source_intervals = _source_interval_fields(derived)
        views.append(
            SequenceViewRecord(
                sequence_id=row_id,
                view_name=_view_name(row_id=row_id, label=labels.get(row_id)),
                aliases=None,
                product_kind="construct_insert",
                context_kind="anchor_only",
                orientation="forward",
                analysis_only=analysis_only,
                source_dataset_id=dataset.name,
                source_label=labels.get(row_id),
                parent_sequence_id=_clean_text(derived.get("derived__parent_id")),
                parent_dataset_id=_clean_text(derived.get("derived__parent_dataset")),
                derivation_id=_clean_text(derived.get("derived__operation")),
                derivation_spec_id=_clean_text(derived.get("derived__spec_id")),
                template_sequence_id=_clean_text(derived.get("derived__template_id")),
                template_dataset_id=_clean_text(derived.get("derived__template_dataset")),
                source_interval_start_0=source_intervals["source_interval_start_0"],
                source_interval_end_0=source_intervals["source_interval_end_0"],
                recommended_pooling="seq_mean",
                created_at=created_at,
                created_by="dnadesign.usr.materialize_promoter_anchor_sequence_views",
            )
        )
    return views


def _existing_view_matches(existing: dict[str, object] | None, view: SequenceViewRecord) -> bool:
    if existing is None:
        return False
    expected = view.model_dump(mode="python")
    expected.pop("created_at", None)
    expected.pop("created_by", None)
    return existing == expected


def materialize_promoter_anchor_sequence_views(
    *,
    usr_root: Path,
    dataset_name: str = DEFAULT_DATASET,
    write: bool = False,
) -> AnchorSequenceViewResult:
    dataset = Dataset(usr_root, dataset_name)
    dataset._require_exists()  # noqa: SLF001
    views = _build_anchor_sequence_views(dataset)
    existing = load_sequence_view_index(dataset)
    missing: list[SequenceViewRecord] = []
    for view in views:
        current = existing.get(str(view.view_id))
        if current is None:
            missing.append(view)
            continue
        if not _existing_view_matches(current, view):
            raise ValueError(f"Existing sequence view '{view.view_id}' has non-idempotent metadata drift.")
    rows_written = 0
    if write and missing:
        rows_written = write_sequence_views(
            dataset,
            [view.model_dump(mode="python") for view in missing],
            conflict_policy="idempotent",
            actor={
                "tool": "usr",
                "run_id": "materialize_promoter_anchor_sequence_views",
                "command": "materialize_promoter_anchor_sequence_views",
                "dataset": dataset.name,
            },
        )
    derived_rows = _read_optional_overlay(dataset, "derived")
    return AnchorSequenceViewResult(
        dataset=dataset.name,
        rows_seen=len(views),
        views_planned=len(views),
        views_written=rows_written,
        existing_views=len(existing),
        analysis_only_views=sum(1 for view in views if view.analysis_only),
        biological_derived_rows=sum(
            1 for row in derived_rows if _clean_text(row.get("derived__product_kind")) == "selected_region"
        ),
        analysis_window_source_rows=sum(
            1 for row in derived_rows if _clean_text(row.get("derived__product_kind")) == "analysis_window"
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize construct_insert sequence views for a merged anchor dataset."
    )
    parser.add_argument("--usr-root", type=Path, default=_default_usr_root())
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--write", action="store_true", help="Write missing sequence-view rows. Default is dry-run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize_promoter_anchor_sequence_views(
        usr_root=args.usr_root,
        dataset_name=args.dataset,
        write=bool(args.write),
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
