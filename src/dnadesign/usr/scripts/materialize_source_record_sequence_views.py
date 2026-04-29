"""
Materialize generic source-record sequence views for USR source datasets.

This helper is intentionally conservative: datasets with existing non-source
sequence-view sidecars keep their domain-specific semantics. Datasets without
sidecars receive one `source_record` view and one mutable view-semantics row per
base record so every non-archived local USR dataset has an explicit semantic
surface.
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
    SequenceViewRecord,
    ViewSemanticsRecord,
    load_sequence_views,
    load_view_semantics_index,
    write_sequence_views,
    write_view_semantics,
)
from dnadesign.usr.src.storage.parquet import now_utc


@dataclass(frozen=True)
class SourceRecordSemanticProfile:
    source_family: str
    selection_basis: str
    view_collections: tuple[str, ...]
    role_tags: tuple[str, ...]
    study_id: str | None = None


@dataclass(frozen=True)
class SourceRecordSidecarResult:
    datasets: list[str]
    datasets_seen: int
    datasets_planned: int
    datasets_written: int
    datasets_existing: int
    views_seen: int
    views_planned: int
    views_written: int
    view_semantics_planned: int
    view_semantics_written: int
    by_dataset: dict[str, dict[str, object]]


DATASET_PROFILES: dict[str, SourceRecordSemanticProfile] = {
    "densegen_demo_sampling_baseline": SourceRecordSemanticProfile(
        source_family="densegen_demo",
        selection_basis="densegen_source_record",
        view_collections=("densegen_demo_sampling_baseline_source_records",),
        role_tags=("source_record", "demo_fixture"),
    ),
    "densegen_prom_eth_cip_source": SourceRecordSemanticProfile(
        source_family="densegen_generated",
        selection_basis="densegen_source_record",
        view_collections=("densegen_prom_eth_cip_source_records", "stress_ethanol_cipro_growth_sources"),
        role_tags=("source_record", "design_population"),
        study_id="stress_ethanol_cipro_growth",
    ),
    "densegen_study_constitutive_sigma_panel": SourceRecordSemanticProfile(
        source_family="densegen_generated",
        selection_basis="densegen_source_record",
        view_collections=("densegen_study_constitutive_sigma_panel_source_records",),
        role_tags=("source_record", "design_population"),
    ),
    "usr_demo_cli_examples": SourceRecordSemanticProfile(
        source_family="demo_fixture",
        selection_basis="demo_source_record",
        view_collections=("usr_demo_cli_examples_source_records",),
        role_tags=("source_record", "demo_fixture"),
    ),
    "usr_mg1655_promoter_controls": SourceRecordSemanticProfile(
        source_family="legacy_reference_control",
        selection_basis="curated_control_source_record",
        view_collections=("usr_mg1655_promoter_controls_source_records", "stress_ethanol_cipro_growth_sources"),
        role_tags=("source_record", "reference_control"),
        study_id="stress_ethanol_cipro_growth",
    ),
    "usr_pdual10_plasmid_template": SourceRecordSemanticProfile(
        source_family="construct_template",
        selection_basis="template_source_record",
        view_collections=("usr_pdual10_plasmid_template_source_records", "stress_ethanol_cipro_growth_sources"),
        role_tags=("source_record", "template_seed"),
        study_id="stress_ethanol_cipro_growth",
    ),
    "usr_sfxi_pdual10_densegen_promoters": SourceRecordSemanticProfile(
        source_family="sfxi_archive",
        selection_basis="archive_backed_source_record",
        view_collections=("usr_sfxi_pdual10_densegen_promoters_source_records", "stress_ethanol_cipro_growth_sources"),
        role_tags=("source_record", "archive_source", "design_population"),
        study_id="stress_ethanol_cipro_growth",
    ),
}


def _default_usr_root() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets"


def _dataset_names_with_records(usr_root: Path) -> list[str]:
    names: list[str] = []
    for records_path in sorted(usr_root.glob("**/records.parquet")):
        rel = records_path.parent.relative_to(usr_root)
        if rel.parts and rel.parts[0] == "archived":
            continue
        names.append(rel.as_posix())
    return names


def _profile_for_dataset(name: str) -> SourceRecordSemanticProfile:
    profile = DATASET_PROFILES.get(name)
    if profile is not None:
        return profile
    if name.startswith("densegen_"):
        return SourceRecordSemanticProfile(
            source_family="densegen_generated",
            selection_basis="densegen_source_record",
            view_collections=(f"{name}_source_records",),
            role_tags=("source_record", "design_population"),
        )
    return SourceRecordSemanticProfile(
        source_family="usr_source",
        selection_basis="source_record",
        view_collections=(f"{name}_source_records",),
        role_tags=("source_record",),
    )


def _record_rows(dataset: Dataset) -> list[dict[str, object]]:
    schema = dataset.schema()
    columns = ["id"]
    for optional in ("source", "usr_label__primary", "usr_label__aliases"):
        if optional in schema.names:
            columns.append(optional)
    rows: list[dict[str, object]] = []
    for batch in dataset.scan(columns=columns, include_overlays=True):
        rows.extend(batch.to_pylist())
    return rows


def _aliases_for_row(row: dict[str, object]) -> list[str] | None:
    aliases: list[str] = []
    primary = row.get("usr_label__primary")
    if primary is not None and str(primary).strip():
        aliases.append(str(primary).strip())
    raw_aliases = row.get("usr_label__aliases")
    if isinstance(raw_aliases, list):
        aliases.extend(str(value).strip() for value in raw_aliases if value is not None and str(value).strip())
    return aliases or None


def _view_name_for_row(dataset: Dataset, row: dict[str, object]) -> str:
    primary = row.get("usr_label__primary")
    if primary is not None and str(primary).strip():
        return f"{str(primary).strip()}_source_record"
    sequence_id = str(row["id"])
    return f"{dataset.name}_source_record_{sequence_id[:12]}"


def _source_record_views(dataset: Dataset, *, created_at: str, created_by: str) -> list[SequenceViewRecord]:
    rows = _record_rows(dataset)
    alias_counts: Counter[str] = Counter()
    for row in rows:
        alias_counts.update(alias.casefold() for alias in _aliases_for_row(row) or [])
    return [
        SequenceViewRecord(
            sequence_id=str(row["id"]),
            view_name=_view_name_for_row(dataset, row),
            aliases=[alias for alias in (_aliases_for_row(row) or []) if alias_counts[alias.casefold()] == 1] or None,
            product_kind="source_record",
            context_kind=None,
            orientation="unknown",
            analysis_only=False,
            source_dataset_id=dataset.name,
            source_label=str(row["usr_label__primary"]).strip()
            if row.get("usr_label__primary") is not None and str(row.get("usr_label__primary")).strip()
            else None,
            recommended_pooling="seq_mean",
            created_at=created_at,
            created_by=created_by,
        )
        for row in rows
    ]


def _view_semantics(
    views: list[SequenceViewRecord],
    *,
    profile: SourceRecordSemanticProfile,
    created_at: str,
    created_by: str,
) -> list[ViewSemanticsRecord]:
    return [
        ViewSemanticsRecord(
            view_id=str(view.view_id),
            sequence_id=str(view.sequence_id),
            source_family=profile.source_family,
            selection_basis=profile.selection_basis,
            view_collections=list(profile.view_collections),
            role_tags=list(profile.role_tags),
            study_id=profile.study_id,
            created_at=created_at,
            created_by=created_by,
        )
        for view in views
    ]


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


def _counts(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values if value is not None).items()))


def _validate_requested_names(names: list[str]) -> None:
    archived = [name for name in names if Path(name).parts and Path(name).parts[0] == "archived"]
    if archived:
        preview = ", ".join(archived)
        raise ValueError(f"Archived datasets are excluded from source-record sidecar materialization: {preview}")


def materialize_source_record_sequence_views(
    *,
    usr_root: Path,
    dataset_names: list[str] | None = None,
    write: bool = False,
) -> SourceRecordSidecarResult:
    names = list(dataset_names or _dataset_names_with_records(usr_root))
    _validate_requested_names(names)
    created_at = now_utc()
    created_by = "dnadesign.usr.materialize_source_record_sequence_views"

    by_dataset: dict[str, dict[str, object]] = {}
    planned_views: list[tuple[Dataset, list[SequenceViewRecord], list[ViewSemanticsRecord]]] = []
    planned_semantics_only: list[tuple[Dataset, list[ViewSemanticsRecord]]] = []
    views_seen = 0
    views_planned = 0
    semantics_planned = 0
    existing_count = 0

    for name in names:
        dataset = Dataset(usr_root, name)
        dataset._require_exists()  # noqa: SLF001
        existing_views = load_sequence_views(dataset)
        existing_semantics = load_view_semantics_index(dataset)
        views_seen += len(existing_views)
        profile = _profile_for_dataset(dataset.name)

        if existing_views:
            missing_semantics = []
            for view in existing_views:
                row = _view_semantics(
                    [view],
                    profile=profile,
                    created_at=created_at,
                    created_by=created_by,
                )[0]
                current = existing_semantics.get(str(row.view_id))
                if current is None:
                    missing_semantics.append(row)
                elif not _matches_existing(current, row) and view.product_kind == "source_record":
                    raise ValueError(
                        f"Existing view-semantics row '{row.view_id}' in dataset '{dataset.name}' has "
                        "non-idempotent metadata drift."
                    )
            if missing_semantics and all(view.product_kind == "source_record" for view in existing_views):
                planned_semantics_only.append((dataset, missing_semantics))
                semantics_planned += len(missing_semantics)
                state = "planned_semantics"
            elif missing_semantics:
                raise ValueError(
                    f"Dataset '{dataset.name}' has non-source sequence views without matching view_semantics rows."
                )
            else:
                existing_count += 1
                state = "existing"
            by_dataset[dataset.name] = {
                "state": state,
                "records": pq.read_metadata(dataset.records_path).num_rows,
                "sequence_views": len(existing_views),
                "view_semantics": len(existing_semantics),
                "product_kinds": _counts([view.product_kind for view in existing_views]),
                "orientations": _counts([view.orientation for view in existing_views]),
                "source_families": _counts([row.get("source_family") for row in existing_semantics.values()]),
                "selection_bases": _counts([row.get("selection_basis") for row in existing_semantics.values()]),
            }
            continue

        views = _source_record_views(dataset, created_at=created_at, created_by=created_by)
        semantics = _view_semantics(views, profile=profile, created_at=created_at, created_by=created_by)
        planned_views.append((dataset, views, semantics))
        views_planned += len(views)
        semantics_planned += len(semantics)
        by_dataset[dataset.name] = {
            "state": "planned",
            "records": pq.read_metadata(dataset.records_path).num_rows,
            "sequence_views": len(views),
            "view_semantics": len(semantics),
            "source_family": profile.source_family,
            "selection_basis": profile.selection_basis,
        }

    views_written = 0
    semantics_written = 0
    written_datasets: set[str] = set()
    if write:
        for dataset, views, semantics in planned_views:
            actor = {
                "tool": "usr",
                "run_id": "materialize_source_record_sequence_views",
                "command": "materialize_source_record_sequence_views",
                "dataset": dataset.name,
            }
            views_written += write_sequence_views(dataset, views, conflict_policy="idempotent", actor=actor)
            semantics_written += write_view_semantics(dataset, semantics, conflict_policy="idempotent", actor=actor)
            written_datasets.add(dataset.name)
            by_dataset[dataset.name]["state"] = "written"
        for dataset, semantics in planned_semantics_only:
            semantics_written += write_view_semantics(
                dataset,
                semantics,
                conflict_policy="idempotent",
                actor={
                    "tool": "usr",
                    "run_id": "materialize_source_record_sequence_views",
                    "command": "materialize_source_record_sequence_views",
                    "dataset": dataset.name,
                },
            )
            written_datasets.add(dataset.name)
            by_dataset[dataset.name]["state"] = "written_semantics"

    return SourceRecordSidecarResult(
        datasets=names,
        datasets_seen=len(names),
        datasets_planned=len(planned_views) + len(planned_semantics_only),
        datasets_written=len(written_datasets),
        datasets_existing=existing_count,
        views_seen=views_seen,
        views_planned=views_planned,
        views_written=views_written,
        view_semantics_planned=semantics_planned,
        view_semantics_written=semantics_written,
        by_dataset=by_dataset,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize generic source-record sequence-view sidecars for non-archived USR datasets."
    )
    parser.add_argument("--usr-root", type=Path, default=_default_usr_root())
    parser.add_argument("--dataset", action="append", dest="datasets", help="Dataset to materialize. Repeatable.")
    parser.add_argument("--write", action="store_true", help="Write missing sidecar rows. Default is dry-run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize_source_record_sequence_views(
        usr_root=args.usr_root,
        dataset_names=args.datasets,
        write=bool(args.write),
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
