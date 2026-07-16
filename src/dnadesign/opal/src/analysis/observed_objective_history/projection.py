"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/observed_objective_history/projection.py

Builds commensurate observed-objective evidence from explicit run snapshots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...core.objective_result import validate_objective_result_v2
from ...core.utils import ExitCodes, OpalError, params_hash
from ...registries.objectives import (
    get_objective,
    get_objective_declared_channels,
    get_objective_observed_replay_contract,
)
from ..ledger import read_run_observed_events, read_runs, require_columns

RUN_SERIES_SCHEMA_VERSION = "opal.observed_objective_run_series.v1"
RUN_CONTRACT_SCHEMA_VERSION = "opal.observed_objective_run_contract.v1"
COMPARABILITY_SCHEMA_VERSION = "opal.observed_objective_comparability.v1"


@dataclass(frozen=True)
class ObservedObjectiveHistory:
    """Observed candidate scores replayed under one commensurate objective contract."""

    frame: pd.DataFrame
    summary: pd.DataFrame
    selection_view_id: str
    objective_name: str
    score_ref: str
    score_channel: str
    objective_mode: str
    y_space: str
    comparability_sha256: str


@dataclass(frozen=True)
class _ResolvedRun:
    as_of_round: int
    run_id: str
    contract_sha256: str
    comparability_sha256: str
    objective_name: str
    objective_params: dict[str, Any]
    score_ref: str
    score_channel: str
    objective_mode: str
    y_space: str
    source_kind: str
    observed_events_sha256: str
    events: pl.DataFrame


def load_observed_objective_history(
    *,
    outputs_dir: Path,
    selection_view_id: str,
    run_series: Mapping[str, object],
) -> ObservedObjectiveHistory:
    """Load observed objective history from one digest-bound run per declared round."""

    outputs_dir = Path(outputs_dir)
    selection_view_id = _canonical_text(selection_view_id, field="selection_view_id")
    entries = _parse_run_series(run_series)
    runs = read_runs(outputs_dir / "ledger" / "runs.parquet")
    resolved: list[_ResolvedRun] = []
    for entry in entries:
        run = _resolve_run(
            runs,
            outputs_dir=outputs_dir,
            selection_view_id=selection_view_id,
            as_of_round=int(entry["as_of_round"]),
            run_id=str(entry["run_id"]),
        )
        expected = str(entry["contract_sha256"])
        if run.contract_sha256 != expected:
            raise OpalError(
                "Observed objective run contract digest mismatch for "
                f"round={run.as_of_round}, run_id={run.run_id!r}: "
                f"expected={expected}, actual={run.contract_sha256}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        resolved.append(run)
    compatibility = {run.comparability_sha256 for run in resolved}
    if len(compatibility) != 1:
        details = [
            {"as_of_round": run.as_of_round, "run_id": run.run_id, "comparability_sha256": run.comparability_sha256}
            for run in resolved
        ]
        raise OpalError(
            "Observed objective run series is not commensurate across objective, Y-space, calibration, "
            f"selection-view, score, or ingest semantics: {details}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    events = _immutable_events(resolved)
    first = resolved[0]
    values = _score_events(
        events,
        objective_name=first.objective_name,
        objective_params=first.objective_params,
        score_channel=first.score_channel,
    )
    events["objective_value"] = values
    events["selection_view_id"] = selection_view_id
    events["objective_name"] = first.objective_name
    events["score_ref"] = first.score_ref
    events["score_channel"] = first.score_channel
    events["objective_mode"] = first.objective_mode
    ordered = events.sort_values(["observed_round", "batch_id", "id"], kind="stable").reset_index(drop=True)
    summary = _batch_summary(ordered, objective_mode=first.objective_mode)
    return ObservedObjectiveHistory(
        frame=ordered,
        summary=summary,
        selection_view_id=selection_view_id,
        objective_name=first.objective_name,
        score_ref=first.score_ref,
        score_channel=first.score_channel,
        objective_mode=first.objective_mode,
        y_space=first.y_space,
        comparability_sha256=first.comparability_sha256,
    )


def _batch_summary(events: pd.DataFrame, *, objective_mode: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = events.groupby(["observed_round", "batch_id"], sort=True, dropna=False)
    for (observed_round, batch_id), batch in grouped:
        values = batch["objective_value"].astype(float)
        cumulative = events.loc[events["observed_round"] <= int(observed_round), "objective_value"].astype(float)
        cumulative_best = float(cumulative.max() if objective_mode == "maximize" else cumulative.min())
        rows.append(
            {
                "observed_round": int(observed_round),
                "batch_id": str(batch_id),
                "candidate_count": int(len(values)),
                "batch_median": float(values.median()),
                "between_candidate_q25": float(values.quantile(0.25)),
                "between_candidate_q75": float(values.quantile(0.75)),
                "cumulative_best": cumulative_best,
            }
        )
    return pd.DataFrame(rows).sort_values(["observed_round", "batch_id"], kind="stable").reset_index(drop=True)


def observed_objective_run_contract_sha256(
    *,
    outputs_dir: Path,
    selection_view_id: str,
    as_of_round: int,
    run_id: str,
) -> str:
    """Compute the digest that pins one run's observed-objective semantics and artifact."""

    outputs_dir = Path(outputs_dir)
    runs = read_runs(outputs_dir / "ledger" / "runs.parquet")
    resolved = _resolve_run(
        runs,
        outputs_dir=outputs_dir,
        selection_view_id=_canonical_text(selection_view_id, field="selection_view_id"),
        as_of_round=_nonnegative_int(as_of_round, field="as_of_round"),
        run_id=_canonical_text(run_id, field="run_id"),
    )
    return resolved.contract_sha256


def _resolve_run(
    runs: pl.DataFrame,
    *,
    outputs_dir: Path,
    selection_view_id: str,
    as_of_round: int,
    run_id: str,
) -> _ResolvedRun:
    require_columns(
        runs,
        (
            "as_of_round",
            "run_id",
            "y_ingest__name",
            "y_ingest__params",
            "objective__defs_json",
            "selection_views__defs_json",
            "artifacts",
        ),
        ctx="observed objective run series",
    )
    scoped = runs.filter((pl.col("as_of_round") == int(as_of_round)) & (pl.col("run_id").cast(pl.Utf8) == str(run_id)))
    if scoped.height != 1:
        raise OpalError(
            "Observed objective run series requires exactly one run row for "
            f"round={as_of_round}, run_id={run_id!r}; found {scoped.height}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    row = scoped.to_dicts()[0]
    objective_defs = _json_definitions(row["objective__defs_json"], field="objective__defs_json")
    selection_defs = _json_definitions(row["selection_views__defs_json"], field="selection_views__defs_json")
    objectives = [item for item in objective_defs if item.get("selection_view_id") == selection_view_id]
    selections = [item for item in selection_defs if item.get("selection_view_id") == selection_view_id]
    if len(objectives) != 1 or len(selections) != 1:
        raise OpalError(
            f"Observed objective selection view {selection_view_id!r} must be defined exactly once in run metadata.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    objective = objectives[0]
    selection = selections[0]
    objective_name = _canonical_text(objective.get("objective_name"), field="objective_name")
    replay_contract = get_objective_observed_replay_contract(objective_name)
    if replay_contract != "pointwise_params_v1":
        raise OpalError(
            f"Objective {objective_name!r} does not declare pointwise observed replay; "
            "history scoring must not infer or fabricate training-state semantics.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    selection_objective_name = _canonical_text(selection.get("objective_name"), field="selection objective_name")
    if selection_objective_name != objective_name:
        raise OpalError(
            "Observed objective selection and objective definitions disagree on objective_name.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    params = objective.get("params")
    if not isinstance(params, Mapping):
        raise OpalError("Observed objective params must be a mapping.", ExitCodes.CONTRACT_VIOLATION)
    objective_params = dict(params)
    selection_params = selection.get("objective_params")
    if not isinstance(selection_params, Mapping) or _contract_hash(dict(selection_params)) != _contract_hash(
        objective_params
    ):
        raise OpalError(
            "Observed objective selection and objective definitions disagree on objective params.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    score_ref = _canonical_text(selection.get("score_ref"), field="score_ref")
    prefix = f"{selection_view_id}/"
    if not score_ref.startswith(prefix) or score_ref == prefix:
        raise OpalError(
            f"Observed objective score_ref must name a channel in selection view {selection_view_id!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    score_channel = score_ref.removeprefix(prefix)
    declared = get_objective_declared_channels(objective_name)
    if score_channel not in declared["score"]:
        raise OpalError(
            f"Observed objective score channel {score_channel!r} is not declared by {objective_name!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    objective_mode = _canonical_text(selection.get("objective_mode"), field="objective_mode").lower()
    declared_mode = declared["score_modes"].get(score_channel)
    if objective_mode != declared_mode:
        raise OpalError(
            f"Observed objective mode mismatch for {score_ref!r}: run={objective_mode!r}, declared={declared_mode!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    score_channels = objective.get("score_channels")
    if (
        not isinstance(score_channels, Sequence)
        or isinstance(score_channels, (str, bytes))
        or score_ref not in score_channels
    ):
        raise OpalError(
            f"Observed objective run metadata does not bind selected score_ref {score_ref!r} to its objective.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    y_ingest_name = _canonical_text(row.get("y_ingest__name"), field="y_ingest__name")
    y_ingest_params = row.get("y_ingest__params")
    if not isinstance(y_ingest_params, Mapping):
        raise OpalError("Observed objective y_ingest__params must be a mapping.", ExitCodes.CONTRACT_VIOLATION)
    observed = read_run_observed_events(
        runs,
        outputs_dir=outputs_dir,
        round_k=as_of_round,
        run_id=run_id,
    )
    y_space = _single_canonical_value(observed.frame, column="y_space", field="Y-space")
    source_kind = _single_canonical_value(observed.frame, column="label_source_kind", field="label-source kind")
    comparability_payload = {
        "schema_version": COMPARABILITY_SCHEMA_VERSION,
        "selection_view_id": selection_view_id,
        "objective_name": objective_name,
        "objective_params": objective_params,
        "observed_replay_contract": replay_contract,
        "score_ref": score_ref,
        "objective_mode": objective_mode,
        "y_ingest": {"name": y_ingest_name, "params": dict(y_ingest_params)},
        "y_space": y_space,
        "label_source_kind": source_kind,
    }
    comparability_sha256 = _contract_hash(comparability_payload)
    contract_payload = {
        "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
        "as_of_round": int(as_of_round),
        "run_id": str(run_id),
        "comparability_sha256": comparability_sha256,
        "observed_events_sha256": observed.sha256,
    }
    return _ResolvedRun(
        as_of_round=int(as_of_round),
        run_id=str(run_id),
        contract_sha256=_contract_hash(contract_payload),
        comparability_sha256=comparability_sha256,
        objective_name=objective_name,
        objective_params=objective_params,
        score_ref=score_ref,
        score_channel=score_channel,
        objective_mode=objective_mode,
        y_space=y_space,
        source_kind=source_kind,
        observed_events_sha256=observed.sha256,
        events=observed.frame,
    )


def _immutable_events(runs: Sequence[_ResolvedRun]) -> pd.DataFrame:
    retained: dict[tuple[str, int], dict[str, Any]] = {}
    fingerprints: dict[tuple[str, int], str] = {}
    for run in runs:
        snapshot_events: dict[tuple[str, int], dict[str, Any]] = {}
        snapshot_fingerprints: dict[tuple[str, int], str] = {}
        for raw in run.events.to_dicts():
            candidate_id = _canonical_text(raw.get("id"), field="event id")
            observed_round = _nonnegative_int(raw.get("observed_round"), field="event observed_round")
            source_kind = _canonical_text(raw.get("label_source_kind"), field=f"event {candidate_id!r} source kind")
            batch_id = _resolved_batch_id(
                raw.get("batch_id"),
                observed_round=observed_round,
                source_kind=source_kind,
            )
            key = (candidate_id, observed_round)
            if key in snapshot_events:
                raise OpalError(
                    f"Observed objective snapshot contains duplicate event key {key!r}.",
                    ExitCodes.CONTRACT_VIOLATION,
                )
            y_obs = _finite_vector(raw.get("y_obs"), field=f"event {key!r} y_obs")
            event = {
                "id": candidate_id,
                "display_label": raw.get("display_label"),
                "sequence": _canonical_text(raw.get("sequence"), field=f"event {key!r} sequence"),
                "observed_round": observed_round,
                "batch_id": batch_id,
                "y_space": _canonical_text(raw.get("y_space"), field=f"event {key!r} y_space"),
                "y_obs": y_obs,
                "label_source_kind": source_kind,
                "evidence_as_of_round": run.as_of_round,
                "evidence_run_id": run.run_id,
                "evidence_observed_events_sha256": run.observed_events_sha256,
            }
            stable_payload = {key: value for key, value in event.items() if not key.startswith("evidence_")}
            fingerprint = _contract_hash(stable_payload)
            snapshot_events[key] = event
            snapshot_fingerprints[key] = fingerprint
        missing_prior = sorted(set(retained) - set(snapshot_events))
        if missing_prior:
            raise OpalError(
                "Observed objective cumulative snapshot drops prior events; "
                f"round={run.as_of_round}, run_id={run.run_id!r}, sample={missing_prior[:10]}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        for key, event in snapshot_events.items():
            fingerprint = snapshot_fingerprints[key]
            if key in fingerprints and fingerprints[key] != fingerprint:
                raise OpalError(
                    f"Observed objective event {key!r} changed across cumulative run snapshots.",
                    ExitCodes.CONTRACT_VIOLATION,
                )
            if key not in retained:
                retained[key] = event
                fingerprints[key] = fingerprint
    if not retained:
        raise OpalError("Observed objective run series contains no immutable events.", ExitCodes.CONTRACT_VIOLATION)
    return pd.DataFrame(retained.values())


def _score_events(
    events: pd.DataFrame,
    *,
    objective_name: str,
    objective_params: Mapping[str, Any],
    score_channel: str,
) -> np.ndarray:
    vectors = events["y_obs"].tolist()
    lengths = {len(vector) for vector in vectors}
    if len(lengths) != 1:
        raise OpalError(
            f"Observed objective events contain inconsistent Y-vector lengths: {sorted(lengths)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    y_obs = np.asarray(vectors, dtype=float)
    objective = get_objective(objective_name)
    try:
        raw = objective(
            y_pred=y_obs,
            params=dict(objective_params),
            ctx=None,
            train_view=None,
            y_pred_std=None,
        )
    except Exception as exc:
        raise OpalError(
            f"Observed objective replay failed for {objective_name!r}: {exc}",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    result = validate_objective_result_v2(result=raw, objective_name=objective_name, n_rows=len(events))
    if score_channel not in result.scores_by_name:
        raise OpalError(
            f"Observed objective replay did not emit selected score channel {score_channel!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return np.asarray(result.scores_by_name[score_channel], dtype=float)


def _json_definitions(raw: object, *, field: str) -> list[dict[str, Any]]:
    try:
        parsed = raw if isinstance(raw, list) else json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OpalError(f"Observed objective {field} is invalid JSON: {exc}", ExitCodes.CONTRACT_VIOLATION) from exc
    if not isinstance(parsed, list) or not all(isinstance(item, Mapping) for item in parsed):
        raise OpalError(
            f"Observed objective {field} must contain a list of mappings.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return [dict(item) for item in parsed]


def _single_canonical_value(frame: pl.DataFrame, *, column: str, field: str) -> str:
    values = frame.get_column(column)
    if values.null_count():
        raise OpalError(f"Observed objective run requires a non-null {field}.", ExitCodes.CONTRACT_VIOLATION)
    unique = values.cast(pl.Utf8).unique().to_list()
    if len(unique) != 1:
        raise OpalError(
            f"Observed objective run requires exactly one {field}; observed={unique!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return _canonical_text(unique[0], field=field)


def _finite_vector(value: object, *, field: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise OpalError(f"Observed objective {field} must be a non-empty numeric vector.", ExitCodes.CONTRACT_VIOLATION)
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise OpalError(f"Observed objective {field} must be numeric.", ExitCodes.CONTRACT_VIOLATION) from exc
    if not np.all(np.isfinite(vector)):
        raise OpalError(f"Observed objective {field} must be finite.", ExitCodes.CONTRACT_VIOLATION)
    return vector.tolist()


def _resolved_batch_id(value: object, *, observed_round: int, source_kind: str) -> str:
    if value is None and source_kind == "campaign_history":
        return f"round-{observed_round}"
    return _canonical_text(value, field="event batch_id")


def _contract_hash(payload: object) -> str:
    try:
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise OpalError(
            f"Observed objective contract payload must be finite, JSON-native data: {exc}",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    return params_hash(payload)


def _parse_run_series(raw: Mapping[str, object]) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, Mapping):
        raise OpalError("Observed objective run_series must be a mapping.", ExitCodes.CONTRACT_VIOLATION)
    allowed = {"schema_version", "runs"}
    extra = sorted(set(raw) - allowed)
    if extra:
        raise OpalError(
            f"Observed objective run_series contains unknown fields: {extra}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if raw.get("schema_version") != RUN_SERIES_SCHEMA_VERSION:
        raise OpalError(
            f"Observed objective run_series schema_version must be {RUN_SERIES_SCHEMA_VERSION!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    runs = raw.get("runs")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)) or not runs:
        raise OpalError(
            "Observed objective run_series.runs must be a non-empty list.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(runs):
        if not isinstance(item, Mapping):
            raise OpalError(
                f"Observed objective run_series.runs[{index}] must be a mapping.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        required = {"as_of_round", "run_id", "contract_sha256"}
        if set(item) != required:
            missing = sorted(required - set(item))
            extra = sorted(set(item) - required)
            raise OpalError(
                "Observed objective run_series entries require exactly "
                f"{sorted(required)}; missing={missing}, extra={extra}. contract_sha256 binds each run's semantics.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        parsed.append(
            {
                "as_of_round": _nonnegative_int(item["as_of_round"], field=f"runs[{index}].as_of_round"),
                "run_id": _canonical_text(item["run_id"], field=f"runs[{index}].run_id"),
                "contract_sha256": _sha256(item["contract_sha256"], field=f"runs[{index}].contract_sha256"),
            }
        )
    rounds = [int(item["as_of_round"]) for item in parsed]
    if rounds != sorted(set(rounds)):
        raise OpalError(
            "Observed objective run_series must declare strictly increasing, unique rounds.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return tuple(parsed)


def _canonical_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OpalError(
            f"Observed objective {field} must be a canonical non-empty string.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return value


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpalError(f"Observed objective {field} must be a nonnegative integer.", ExitCodes.CONTRACT_VIOLATION)
    return int(value)


def _sha256(value: object, *, field: str) -> str:
    digest = _canonical_text(value, field=field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise OpalError(f"Observed objective {field} must be a lowercase SHA-256 digest.", ExitCodes.CONTRACT_VIOLATION)
    return digest


__all__ = [
    "ObservedObjectiveHistory",
    "RUN_SERIES_SCHEMA_VERSION",
    "load_observed_objective_history",
    "observed_objective_run_contract_sha256",
]
