"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/measured_reader_vec8/staging.py

Builds measured Reader vec8 batch0 inputs for stress OPAL campaigns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    BATCH0_HANDOFF_ID,
    OPAL_INGEST_COLUMNS,
    POST_LABEL_ACTIVE_SELECTION_TOP_K_PER_CAMPAIGN,
    READER_EVIDENCE_FILENAME,
    READER_EVIDENCE_PLOT_LABELS,
    READER_EVIDENCE_SCHEMA_VERSION,
    READER_VEC8_RECORD_ID,
    ROUND0_OBSERVED_LABEL_POOL_ID,
    ROUND0_OBSERVED_LABEL_ROLE,
    STRESS_CAMPAIGN_SLUGS,
    TARGET_TIME_H,
    X_COLUMN,
)
from .contracts import MeasuredReaderVec8Error, MeasuredReaderVec8Staging, MeasuredReaderVec8WriteResult, now_iso
from .reader_sources import load_reader_vec8_records
from .sequence_sources import (
    EXCLUDED_READER_DESIGNS,
    build_reader_sequence_resolution_table,
    candidate_x_status,
    load_reader_sequence_resolution_sources,
)
from .synthesis import load_batch0_manifest_map


def build_measured_reader_vec8_staging(*, repo_root: Path, reader_root: Path) -> MeasuredReaderVec8Staging:
    repo = Path(repo_root).expanduser().resolve()
    reader_rows, sources = load_reader_vec8_records(Path(reader_root))
    sequence_resolution = build_reader_sequence_resolution_table(repo)
    source_tables = load_reader_sequence_resolution_sources(repo)
    candidate_records = source_tables["candidate_records"]
    manifests = load_batch0_manifest_map(repo)

    manifest_cols = {
        "reader_design_id": "design_id",
        "id": "candidate_id",
        "core_sequence": "manifest_sequence",
    }
    manifest_map = manifests.rename(columns=manifest_cols)
    manifest_map = manifest_map.loc[
        :,
        ["design_id", "candidate_id", "synthesis_name", "campaign_slug", "manifest_sequence"],
    ]
    sequence_resolution = sequence_resolution.rename(
        columns={
            "reader_design_id": "design_id",
            "sequence": "sequence_from_resolution",
        }
    )
    resolved = reader_rows.rename(columns={"sequence": "sequence_in_reader"}).merge(
        manifest_map, on="design_id", how="left", validate="many_to_one"
    )
    resolved = resolved.merge(
        sequence_resolution,
        on="design_id",
        how="left",
        validate="many_to_one",
    )

    resolved["resolved_sequence"] = resolved["manifest_sequence"].combine_first(resolved["sequence_from_resolution"])
    resolved["sequence_status"] = np.where(resolved["resolved_sequence"].notna(), "resolved", "missing")
    excluded = resolved["design_id"].astype(str).isin(EXCLUDED_READER_DESIGNS)
    resolved.loc[excluded, "sequence_status"] = "explicitly_excluded_from_x_onboarding"
    resolved.loc[excluded & resolved["resolved_sequence"].isna(), "resolved_sequence"] = resolved.loc[
        excluded & resolved["resolved_sequence"].isna(), "sequence_in_reader"
    ]
    resolved = _assign_candidate_identity(resolved, candidate_records)
    resolved["campaign_role"] = _campaign_role(resolved)

    candidate_rows = resolved.loc[resolved["campaign_role"].eq(ROUND0_OBSERVED_LABEL_ROLE)].copy()
    _validate_candidate_rows(candidate_rows, candidate_records)
    measured, duplicates = _deduplicate_candidate_rows(candidate_rows)
    return MeasuredReaderVec8Staging(
        audit_frame=resolved.reset_index(drop=True),
        measured_frame=measured.reset_index(drop=True),
        duplicate_frame=duplicates.reset_index(drop=True),
        source_records=sources,
    )


def _candidate_x_scope(candidate_records: pd.DataFrame, candidate_id: str) -> str:
    if candidate_x_status(candidate_records, candidate_id) == "available":
        return "current_candidate_table"
    return "missing"


def _assign_candidate_identity(frame: pd.DataFrame, candidate_records: pd.DataFrame) -> pd.DataFrame:
    resolved = frame.copy()
    candidate_lookup = candidate_records.loc[:, ["id", "sequence"]].rename(
        columns={
            "id": "sequence_source_id",
            "sequence": "candidate_sequence_from_source",
        }
    )
    candidate_lookup["sequence_source_id"] = candidate_lookup["sequence_source_id"].astype(str)
    resolved["sequence_source_id"] = resolved["sequence_source_id"].astype("string")
    resolved = resolved.merge(candidate_lookup, on="sequence_source_id", how="left", validate="many_to_one")

    manifest_mask = resolved["candidate_id"].notna()
    resolved.loc[manifest_mask, "sequence_source"] = "batch0_synthesis_manifest"
    resolved.loc[manifest_mask, "sequence_source_id"] = resolved.loc[manifest_mask, "candidate_id"].astype(str)

    source_candidate_mask = (
        resolved["candidate_id"].isna()
        & resolved["sequence_source_id"].notna()
        & resolved["candidate_sequence_from_source"].notna()
        & resolved["resolved_sequence"].notna()
        & resolved["resolved_sequence"].astype(str).eq(resolved["candidate_sequence_from_source"].astype(str))
    )
    resolved.loc[source_candidate_mask, "candidate_id"] = resolved.loc[
        source_candidate_mask, "sequence_source_id"
    ].astype(str)
    resolved.loc[source_candidate_mask & resolved["synthesis_name"].isna(), "synthesis_name"] = (
        "reader:" + resolved.loc[source_candidate_mask & resolved["synthesis_name"].isna(), "design_id"].astype(str)
    )
    resolved.loc[source_candidate_mask & resolved["campaign_slug"].isna(), "campaign_slug"] = (
        "shared_reader_vec8_round0"
    )

    candidate_mask = resolved["candidate_id"].notna()
    resolved.loc[candidate_mask, "x_scope"] = resolved.loc[candidate_mask, "candidate_id"].map(
        lambda candidate_id: _candidate_x_scope(candidate_records, candidate_id)
    )
    return resolved


def write_measured_reader_vec8_batch0(
    *,
    repo_root: Path,
    reader_root: Path,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> MeasuredReaderVec8WriteResult:
    repo = Path(repo_root).expanduser().resolve()
    staging = build_measured_reader_vec8_staging(repo_root=repo, reader_root=reader_root)
    output_dir = (
        Path(out_dir).expanduser().resolve()
        if out_dir is not None
        else repo / "src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/measured_reader_vec8"
    )
    audit_csv = output_dir / "reader_vec8_superset_audit.csv"
    manifest_json = output_dir / "reader_vec8_superset_manifest.json"
    campaign_inputs = {
        str(slug): repo / "src/dnadesign/opal/campaigns" / str(slug) / "inputs" / "r0" / "reader_vec8_batch0.csv"
        for slug in STRESS_CAMPAIGN_SLUGS
    }
    campaign_evidence_manifests = {
        slug: path.parent / READER_EVIDENCE_FILENAME for slug, path in campaign_inputs.items()
    }

    targets = [audit_csv, manifest_json, *campaign_inputs.values(), *campaign_evidence_manifests.values()]
    _check_targets(targets, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in campaign_inputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    manifest_payload = _manifest_payload(
        staging,
        audit_csv=audit_csv,
        campaign_inputs=campaign_inputs,
        campaign_evidence_manifests=campaign_evidence_manifests,
    )
    try:
        with TemporaryDirectory(prefix=".reader-vec8-stage.", dir=output_dir) as tmp_raw:
            tmp_dir = Path(tmp_raw)
            tmp_audit = tmp_dir / audit_csv.name
            tmp_manifest = tmp_dir / manifest_json.name
            staging.audit_frame.to_csv(tmp_audit, index=False)
            tmp_manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            tmp_campaigns: dict[str, Path] = {}
            tmp_evidence_manifests: dict[str, Path] = {}
            for slug, target in campaign_inputs.items():
                tmp_path = tmp_dir / f"{slug}__{target.name}"
                _campaign_ingest_frame(staging.measured_frame, campaign_slug=slug).to_csv(tmp_path, index=False)
                tmp_campaigns[slug] = tmp_path
                tmp_evidence = tmp_dir / f"{slug}__{READER_EVIDENCE_FILENAME}"
                tmp_evidence.write_text(
                    json.dumps(
                        _reader_evidence_payload(
                            staging,
                            campaign_slug=slug,
                            label_input=target,
                        ),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                tmp_evidence_manifests[slug] = tmp_evidence

            _replace(tmp_audit, audit_csv)
            _replace(tmp_manifest, manifest_json)
            for slug, tmp_path in tmp_campaigns.items():
                _replace(tmp_path, campaign_inputs[slug])
            for slug, tmp_path in tmp_evidence_manifests.items():
                _replace(tmp_path, campaign_evidence_manifests[slug])
    except Exception as exc:
        if isinstance(exc, MeasuredReaderVec8Error):
            raise
        raise MeasuredReaderVec8Error(f"Failed to write measured reader vec8 staging artifacts: {exc}") from exc

    return MeasuredReaderVec8WriteResult(
        staging=staging,
        audit_csv=audit_csv,
        manifest_json=manifest_json,
        campaign_inputs=campaign_inputs,
        campaign_evidence_manifests=campaign_evidence_manifests,
    )


def _campaign_role(frame: pd.DataFrame) -> pd.Series:
    roles = pd.Series(["unresolved"] * len(frame), index=frame.index, dtype="object")
    candidate_mask = (
        frame["candidate_id"].notna()
        & frame["x_scope"].eq("current_candidate_table")
        & frame["reference_design_id"].astype(str).eq("pDual-10")
    )
    roles.loc[candidate_mask] = ROUND0_OBSERVED_LABEL_ROLE
    roles.loc[frame["sequence_source"].eq("usr_promoter_references")] = "control_reference"
    roles.loc[frame["sequence_source"].eq("usr_sfxi_pdual10_densegen_promoters")] = "historical_reference"
    roles.loc[candidate_mask] = ROUND0_OBSERVED_LABEL_ROLE
    roles.loc[frame["design_id"].astype(str).isin(EXCLUDED_READER_DESIGNS)] = "reader_only_excluded"
    return roles


def _validate_candidate_rows(frame: pd.DataFrame, candidate_records: pd.DataFrame) -> None:
    if frame.empty:
        raise MeasuredReaderVec8Error("Measured reader vec8 staging found no campaign candidate rows.")
    candidate = candidate_records.set_index("id")
    problems: list[str] = []
    for _, row in frame.iterrows():
        candidate_id = str(row["candidate_id"])
        if candidate_id not in candidate.index:
            problems.append(f"{candidate_id}: missing candidate")
            continue
        candidate_sequence = str(candidate.loc[candidate_id, "sequence"])
        if str(row["resolved_sequence"]) != candidate_sequence:
            problems.append(f"{candidate_id}: sequence mismatch")
        if X_COLUMN not in candidate_records.columns or not _present_vector(candidate.loc[candidate_id, X_COLUMN]):
            problems.append(f"{candidate_id}: missing X value")
    if problems:
        raise MeasuredReaderVec8Error("Measured reader vec8 candidate validation failed: " + "; ".join(problems[:10]))


def _deduplicate_candidate_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = frame.copy()
    ranked["time_error_h"] = (pd.to_numeric(ranked["time_selected_h"], errors="coerce") - TARGET_TIME_H).abs()
    ranked["experiment_date"] = (
        ranked["reader_experiment_id"].astype(str).str.extract(r"^(\d{8})", expand=False).fillna("")
    )
    sort_columns = [
        "candidate_id",
        "time_error_h",
        "experiment_date",
        "reader_vec8_table_path",
        "reader_source_row_index",
    ]
    ranked = ranked.sort_values(
        sort_columns,
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    duplicate_mask = ranked.duplicated(subset=["candidate_id"], keep="first")
    measured = ranked.loc[~duplicate_mask].copy()
    duplicates = ranked.loc[duplicate_mask].copy()
    measured["dedup_policy"] = "nearest_12h_then_newest_experiment"
    if not duplicates.empty:
        duplicates["dedup_policy"] = "dropped_nearest_12h_then_newest_experiment"
    return measured, duplicates


def _campaign_ingest_frame(frame: pd.DataFrame, *, campaign_slug: str) -> pd.DataFrame:
    selected = frame.copy()
    selected["campaign_input_slug"] = str(campaign_slug)
    selected["id"] = selected["candidate_id"].astype(str)
    selected["sequence"] = selected["resolved_sequence"].astype(str)
    missing = [column for column in OPAL_INGEST_COLUMNS if column not in selected.columns]
    if missing:
        raise MeasuredReaderVec8Error(f"Measured reader vec8 campaign input missing columns before write: {missing}")
    return selected.loc[:, list(OPAL_INGEST_COLUMNS)].sort_values("design_id", kind="stable").reset_index(drop=True)


def _reader_evidence_payload(
    staging: MeasuredReaderVec8Staging,
    *,
    campaign_slug: str,
    label_input: Path,
) -> dict[str, Any]:
    source_by_experiment = {source.experiment_id: source for source in staging.source_records}
    rows = _reader_evidence_rows(staging.measured_frame, source_by_experiment=source_by_experiment)
    reader_experiments = sorted({str(row["reader_experiment_id"]) for row in rows})
    artifact_count = sum(len(row["artifacts"]) for row in rows)
    return {
        "schema_version": READER_EVIDENCE_SCHEMA_VERSION,
        "created_at": now_iso(),
        "campaign_slug": str(campaign_slug),
        "round": "r0",
        "observed_round": 0,
        "label_input": str(label_input),
        "summary": {
            "rows": len(rows),
            "distinct_ids": len({str(row["id"]) for row in rows}),
            "reader_experiments": len(reader_experiments),
            "artifact_count": artifact_count,
            "missing_artifact_rows": sum(1 for row in rows if row["missing_artifact_kinds"]),
        },
        "rows": rows,
    }


def _reader_evidence_rows(frame: pd.DataFrame, *, source_by_experiment: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.sort_values("design_id", kind="stable").iterrows():
        design_id = str(row["design_id"])
        artifacts = _reader_artifacts_for_row(row, source_by_experiment=source_by_experiment)
        expected = {"reader_vec8_table", *READER_EVIDENCE_PLOT_LABELS.values()}
        present = {str(artifact["semantic_kind"]) for artifact in artifacts if artifact.get("exists")}
        rows.append(
            {
                "id": str(row["candidate_id"]),
                "sequence": str(row["resolved_sequence"]),
                "design_id": design_id,
                "synthesis_name": _optional_str(row.get("synthesis_name")),
                "reader_experiment_id": str(row["reader_experiment_id"]),
                "time_selected_h": float(row["time_selected_h"]),
                "intensity_log2_offset_delta": float(row["intensity_log2_offset_delta"]),
                "reader_config_path": str(row["reader_config_path"]),
                "reader_record_id": READER_VEC8_RECORD_ID,
                "reader_vec8_table_path": str(row["reader_vec8_table_path"]),
                "artifacts": artifacts,
                "missing_artifact_kinds": sorted(expected - present),
            }
        )
    return rows


def _reader_artifacts_for_row(row: pd.Series, *, source_by_experiment: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = [_reader_vec8_table_artifact(row)]
    source = _source_for_row(row, source_by_experiment=source_by_experiment)
    if source is None:
        return artifacts
    design_id = str(row["design_id"])
    aliases = _reader_plot_aliases(design_id, _optional_str(row.get("synthesis_name")))
    for record_id, semantic_kind in READER_EVIDENCE_PLOT_LABELS.items():
        files = source.plot_files_by_record_id.get(record_id, ())
        selected = _select_reader_plot_file(record_id, files, aliases)
        if selected is None:
            continue
        path = _reader_output_path(source.config_path.parent, selected)
        artifacts.append(
            {
                "semantic_kind": semantic_kind,
                "kind": "reader_plot",
                "record_id": record_id,
                "scope": "experiment" if semantic_kind == "sfxi_vec8_heatmap" else "design",
                "path": str(path),
                "path_label": f"{source.experiment_id}/outputs/{selected}",
                "exists": path.exists(),
                "media_type": _media_type(path),
            }
        )
    return artifacts


def _reader_vec8_table_artifact(row: pd.Series) -> dict[str, Any]:
    path = Path(str(row["reader_vec8_table_path"]))
    return {
        "semantic_kind": "reader_vec8_table",
        "kind": "reader_record",
        "record_id": READER_VEC8_RECORD_ID,
        "scope": "design",
        "path": str(path),
        "path_label": f"{row['reader_experiment_id']}/outputs/{_output_relative_path(path, row)}",
        "exists": path.exists(),
        "media_type": "application/vnd.apache.parquet",
    }


def _source_for_row(row: pd.Series, *, source_by_experiment: dict[str, Any]):
    return source_by_experiment.get(str(row["reader_experiment_id"]))


def _reader_plot_aliases(design_id: str, synthesis_name: str | None) -> tuple[str, ...]:
    aliases = {design_id}
    if synthesis_name:
        aliases.add(synthesis_name)
        if synthesis_name.startswith("SECG-B0-"):
            aliases.add(synthesis_name.removeprefix("SECG-B0-"))
    if design_id.startswith("pDual-10-SECG-B0-"):
        aliases.add(design_id.removeprefix("pDual-10-SECG-B0-"))
    if design_id.startswith("pDual-10-"):
        aliases.add(design_id.removeprefix("pDual-10-"))
    if design_id == "pDual-10":
        aliases.add("J23105")
    if "spyp" in aliases:
        aliases.add("spyP")
    return tuple(sorted(alias for alias in aliases if alias))


def _select_reader_plot_file(record_id: str, files: tuple[str, ...], aliases: tuple[str, ...]) -> str | None:
    if record_id == "plot:sfxi_vec8_heatmap":
        return files[0] if files else None
    for alias in aliases:
        raw_candidates = (
            f"ts_{alias}",
            f"ts_snap_YFP_CFP_design_id_alias_{alias}",
            f"ts_snap_YFP_OD600_design_id_alias_{alias}",
            f"snap_ch_{alias}",
        )
        for file_path in files:
            if Path(file_path).stem in raw_candidates:
                return file_path
    return None


def _reader_output_path(experiment_dir: Path, output_file: str) -> Path:
    raw = Path(str(output_file))
    return raw if raw.is_absolute() else (experiment_dir / "outputs" / raw).resolve()


def _output_relative_path(path: Path, row: pd.Series) -> str:
    try:
        experiment_dir = Path(str(row["reader_config_path"])).parent
        return path.resolve().relative_to((experiment_dir / "outputs").resolve()).as_posix()
    except Exception:
        return path.name


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".csv":
        return "text/csv"
    return "application/octet-stream"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _manifest_payload(
    staging: MeasuredReaderVec8Staging,
    *,
    audit_csv: Path,
    campaign_inputs: dict[str, Path],
    campaign_evidence_manifests: dict[str, Path],
) -> dict[str, Any]:
    measured_count = int(len(staging.measured_frame))
    return {
        "schema_version": "stress_ethanol_cipro_growth.measured_reader_vec8.v1",
        "created_at": now_iso(),
        "summary": staging.summary,
        "target_time_h": TARGET_TIME_H,
        "dedup_policy": "nearest_12h_then_newest_experiment",
        "audit_csv": str(audit_csv),
        "campaign_inputs": {slug: str(path) for slug, path in sorted(campaign_inputs.items())},
        "campaign_evidence_manifests": {slug: str(path) for slug, path in sorted(campaign_evidence_manifests.items())},
        "measured_rows_per_campaign_input": {slug: measured_count for slug in sorted(campaign_inputs)},
        "round0_observed_label_pool": {
            "id": ROUND0_OBSERVED_LABEL_POOL_ID,
            "role": "campaign_shared_observed_label_input",
            "rows_per_campaign_input": measured_count,
            "campaign_inputs_are_identical": True,
            "requires_existing_candidate_id_sequence_and_x": True,
            "reference_anchor_design_id": "pDual-10",
        },
        "batch0_synthesis_seed": {
            "handoff_id": BATCH0_HANDOFF_ID,
            "role": "physical_pre_assay_seed_order",
            "does_not_constrain_round0_observed_label_pool": True,
        },
        "post_label_active_selection": {
            "role": "future_model_scored_active_learning_selection",
            "top_k_per_campaign": POST_LABEL_ACTIVE_SELECTION_TOP_K_PER_CAMPAIGN,
            "pooled_campaign_count": len(campaign_inputs),
        },
        "source_rows_by_campaign_slug": {
            str(key): int(value)
            for key, value in staging.measured_frame.groupby("campaign_slug").size().to_dict().items()
        },
        "reader_sources": [source.to_dict() for source in staging.source_records],
    }


def _check_targets(paths: list[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise MeasuredReaderVec8Error(
            "Measured reader vec8 staging outputs already exist; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing[:10])
        )
    bad = [path for path in existing if not path.is_file()]
    if bad:
        raise MeasuredReaderVec8Error(
            "Measured reader vec8 staging target exists but is not a file: " + ", ".join(map(str, bad))
        )


def _replace(tmp_path: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(target)


def _present_vector(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    try:
        return len(value) > 0
    except TypeError:
        return True
