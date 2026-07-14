"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/reader_response_bundle.py

Consume the public Reader response-window bundle without Reader imports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .reader_display_contract import READER_DISPLAY_SCHEMA, response_example_labels, validate_reader_display

READER_BUNDLE_SCHEMA = "reader.response_window.bundle.v4"
EXPECTED_CONTRACTS = {
    "wells": "plate_reader.response_window.wells.v2",
    "designs": "plate_reader.response_window.designs.v2",
    "bootstrap_draws": "plate_reader.response_window.bootstrap_draws.v2",
    "traces": "plate_reader.response_window.traces.v2",
    "events": "plate_reader.response_window.events.v2",
}
EXPECTED_RECORD_ARTIFACTS = {record_id: f"tables/{record_id}.parquet" for record_id in EXPECTED_CONTRACTS}
STATE_ORDER = ("00", "10", "01", "11")
VALUE_COLUMNS = tuple(f"r{state}" for state in STATE_ORDER) + tuple(f"b{state}" for state in STATE_ORDER)
CANDIDATE_IDENTITY_COLUMNS = ("id", "design_id", "reader_experiment_id")


@dataclass(frozen=True)
class ReaderResponseBundle:
    root: Path
    manifest_path: Path
    manifest: dict[str, object]
    designs: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    events: pd.DataFrame

    @property
    def primary_reduction_id(self) -> str:
        value = self.manifest.get("primary_reduction_id")
        if not isinstance(value, str) or not value:
            raise ValueError("Reader bundle lacks primary_reduction_id.")
        return value

    @property
    def response_examples(self) -> dict[str, str]:
        return response_example_labels(self.manifest.get("display"))


def load_reader_response_bundle(path: Path, *, expected_request_path: Path) -> ReaderResponseBundle:
    root = Path(path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Reader response-window manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != READER_BUNDLE_SCHEMA:
        raise ValueError(f"Reader response-window bundle must use {READER_BUNDLE_SCHEMA!r}.")
    if payload.get("study_id") != "stress_ethanol_cipro_growth":
        raise ValueError("Reader response-window bundle study identity disagrees with this study.")
    contracts = payload.get("contracts")
    if contracts != EXPECTED_CONTRACTS:
        raise ValueError(f"Reader response-window contracts disagree: {contracts!r}.")
    if payload.get("state_order") != list(STATE_ORDER):
        raise ValueError(f"Reader response-window state order must be {list(STATE_ORDER)!r}.")
    validate_reader_display(payload.get("display"))
    request_path = Path(expected_request_path).expanduser().resolve()
    if not request_path.is_file():
        raise FileNotFoundError(f"expected Reader response-window request not found: {request_path}")
    request = payload.get("request")
    if (
        not isinstance(request, dict)
        or request.get("artifact_id") != "request.yaml"
        or request.get("sha256") != _sha256(request_path)
    ):
        raise ValueError("Reader response-window request digest disagrees with the checked-in study request.")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Reader response-window bundle has no artifact inventory.")
    for artifact_id, raw in artifacts.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Reader artifact {artifact_id!r} metadata must be a mapping.")
        relative = raw.get("path")
        expected = raw.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"Reader artifact {artifact_id!r} lacks path or digest.")
        artifact_path = (root / relative).resolve()
        try:
            artifact_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Reader artifact {artifact_id!r} escapes the bundle root.") from exc
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Reader artifact {artifact_id!r} is missing: {artifact_path}")
        if _sha256(artifact_path) != expected:
            raise ValueError(f"Reader artifact {artifact_id!r} digest mismatch.")

    records = payload.get("records")
    expected_records = {
        record_id: {"contract_id": EXPECTED_CONTRACTS[record_id], "artifact_id": artifact_id}
        for record_id, artifact_id in EXPECTED_RECORD_ARTIFACTS.items()
    }
    if records != expected_records:
        raise ValueError(f"Reader response-window record map disagrees: {records!r}.")
    missing_record_artifacts = sorted(set(EXPECTED_RECORD_ARTIFACTS.values()) - set(artifacts))
    if missing_record_artifacts:
        raise ValueError(f"Reader response-window bundle lacks record artifacts: {missing_record_artifacts}.")
    designs = pd.read_parquet(_record_path(root, records, record_id="designs"))
    draws = pd.read_parquet(_record_path(root, records, record_id="bootstrap_draws"))
    events = pd.read_parquet(_record_path(root, records, record_id="events"))
    _validate_bundle_frames(designs=designs, draws=draws, events=events, payload=payload)
    return ReaderResponseBundle(
        root=root,
        manifest_path=manifest_path,
        manifest=payload,
        designs=designs,
        bootstrap_draws=draws,
        events=events,
    )


def build_selected_response_labels(
    bundle: ReaderResponseBundle,
    *,
    candidate_identity_bindings: pd.DataFrame,
) -> pd.DataFrame:
    if missing := sorted(set(CANDIDATE_IDENTITY_COLUMNS) - set(candidate_identity_bindings.columns)):
        raise ValueError(f"candidate identity bindings lack Reader identity fields: {missing}")
    labels = candidate_identity_bindings.loc[:, CANDIDATE_IDENTITY_COLUMNS].copy()
    labels["id"] = labels["id"].astype(str)
    labels["design_id"] = labels["design_id"].astype(str)
    labels["reader_experiment_id"] = labels["reader_experiment_id"].astype(str)
    if labels["id"].duplicated().any():
        raise ValueError("candidate identity bindings contain duplicate candidate ids.")
    designs = bundle.designs.loc[~bundle.designs["is_reference"].astype(bool)].copy()
    designs = designs.rename(
        columns={
            "experiment_id": "reader_experiment_id",
            "reduction_role": "screen_role",
        }
    )
    selected = labels.merge(
        designs,
        on=["reader_experiment_id", "design_id"],
        how="left",
        validate="one_to_many",
    )
    if selected["reduction_id"].isna().any():
        missing_rows = selected.loc[selected["reduction_id"].isna(), ["id", "reader_experiment_id", "design_id"]]
        raise ValueError(f"Reader response-window bundle lacks selected labels: {missing_rows.to_dict('records')}")
    expected_rows = len(labels) * bundle.designs["reduction_id"].nunique()
    if len(selected) != expected_rows:
        raise ValueError(f"selected Reader label rows expected {expected_rows}; observed {len(selected)}.")
    if selected.duplicated(subset=["reduction_id", "id"]).any():
        raise ValueError("selected Reader labels contain duplicate reduction/candidate identities.")
    return selected.sort_values(["reduction_id", "id"], kind="mergesort").reset_index(drop=True)


def build_selected_bootstrap_draws(
    bundle: ReaderResponseBundle,
    *,
    candidate_identity_bindings: pd.DataFrame,
) -> pd.DataFrame:
    if missing := sorted(set(CANDIDATE_IDENTITY_COLUMNS) - set(candidate_identity_bindings.columns)):
        raise ValueError(f"candidate identity bindings lack Reader identity fields: {missing}")
    labels = candidate_identity_bindings.loc[:, CANDIDATE_IDENTITY_COLUMNS].copy()
    labels["reader_experiment_id"] = labels["reader_experiment_id"].astype(str)
    labels["design_id"] = labels["design_id"].astype(str)
    draws = bundle.bootstrap_draws.loc[~bundle.bootstrap_draws["is_reference"].astype(bool)].rename(
        columns={"experiment_id": "reader_experiment_id"}
    )
    selected = labels.merge(
        draws,
        on=["reader_experiment_id", "design_id"],
        how="left",
        validate="one_to_many",
    )
    if selected["draw_index"].isna().any():
        raise ValueError("Reader bootstrap bundle lacks one or more selected labels.")
    key = ["id", "reduction_id", "draw_index"]
    if selected.duplicated(subset=key).any():
        raise ValueError("selected Reader bootstrap draws contain duplicate identities.")
    return selected.sort_values(key, kind="mergesort").reset_index(drop=True)


def build_all_primary_measurements(bundle: ReaderResponseBundle) -> pd.DataFrame:
    result = bundle.designs.loc[
        bundle.designs["reduction_id"].astype(str).eq(bundle.primary_reduction_id)
        & ~bundle.designs["is_reference"].astype(bool)
    ].copy()
    result = result.rename(columns={"experiment_id": "reader_experiment_id"})
    result["id"] = result["reader_experiment_id"].astype(str) + "::" + result["design_id"].astype(str)
    return result


def _validate_bundle_frames(
    *,
    designs: pd.DataFrame,
    draws: pd.DataFrame,
    events: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    design_required = {"experiment_id", "design_id", "reduction_id", "reduction_role", "is_reference", *VALUE_COLUMNS}
    draw_required = {"experiment_id", "design_id", "reduction_id", "draw_index", "is_reference", *VALUE_COLUMNS}
    event_required = {
        "experiment_id",
        "event_id",
        "event_interval_start_assay_h",
        "event_interval_end_assay_h",
        "event_time_estimate_assay_h",
    }
    for label, frame, required in (
        ("designs", designs, design_required),
        ("bootstrap_draws", draws, draw_required),
        ("events", events, event_required),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Reader {label} record lacks columns: {missing}")
    if designs.duplicated(subset=["experiment_id", "design_id", "reduction_id"]).any():
        raise ValueError("Reader design identities are not unique.")
    if draws.duplicated(subset=["experiment_id", "design_id", "reduction_id", "draw_index"]).any():
        raise ValueError("Reader bootstrap-draw identities are not unique.")
    if events["experiment_id"].duplicated().any():
        raise ValueError("Reader event identities are not unique.")
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Reader bundle lacks row counts.")
    expected_counts = {
        "design_rows": len(designs),
        "bootstrap_draw_rows": len(draws),
        "experiments": len(events),
    }
    for key, observed in expected_counts.items():
        if int(counts.get(key, -1)) != observed:
            raise ValueError(f"Reader bundle count mismatch for {key}: {counts.get(key)!r} != {observed}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _record_path(root: Path, records: object, *, record_id: str) -> Path:
    if not isinstance(records, dict):
        raise ValueError("Reader response-window bundle lacks a record map.")
    raw = records.get(record_id)
    if not isinstance(raw, dict) or not isinstance(raw.get("artifact_id"), str):
        raise ValueError(f"Reader response-window record {record_id!r} lacks an artifact id.")
    path = (root / str(raw["artifact_id"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Reader response-window record {record_id!r} escapes the bundle root.") from exc
    return path


__all__ = [
    "EXPECTED_CONTRACTS",
    "EXPECTED_RECORD_ARTIFACTS",
    "READER_BUNDLE_SCHEMA",
    "READER_DISPLAY_SCHEMA",
    "ReaderResponseBundle",
    "build_all_primary_measurements",
    "build_selected_bootstrap_draws",
    "build_selected_response_labels",
    "load_reader_response_bundle",
]
