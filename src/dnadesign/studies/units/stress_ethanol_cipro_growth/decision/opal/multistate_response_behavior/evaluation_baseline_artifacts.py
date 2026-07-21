"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/evaluation_baseline_artifacts.py

Artifact verification for the round-0 MSRB evaluation baseline.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from dnadesign.opal import (
    SELECTION_ALLOCATION_PREVIEW_API_VERSION,
    preview_round_robin_next_best_unallocated,
    score_multistate_response_behavior,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    load_study_promoter_alias_registry,
)

from .evaluation_baseline_contracts import (
    CAMPAIGN_SLUG,
    RUN_ID,
    FrozenArtifact,
    FrozenFile,
    MsrbEvaluationBaselineError,
    ParsedBaseline,
    SelectionReplayEvidence,
)


def load_frozen_artifact(
    value: object,
    *,
    root: Path,
    artifact_id: str,
    expected_count: int,
) -> FrozenArtifact:
    """Parse one artifact reference and verify its bytes and declared row count."""

    raw = _mapping(value, context=f"artifacts.{artifact_id}")
    _exact_fields(raw, {"path", "sha256", "row_count"}, context=f"artifacts.{artifact_id}")
    relative_path = _relative_path(raw["path"], context=f"artifacts.{artifact_id}.path")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MsrbEvaluationBaselineError(f"{artifact_id} path escapes the repository.") from exc
    if not path.is_file():
        raise MsrbEvaluationBaselineError(f"{artifact_id} artifact is missing: {path}")
    expected_sha256 = _sha256_text(raw["sha256"], context=f"artifacts.{artifact_id}.sha256")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise MsrbEvaluationBaselineError(
            f"{artifact_id} SHA-256 mismatch: expected {expected_sha256}, observed {actual_sha256}."
        )
    row_count = _positive_integer(raw["row_count"], context=f"artifacts.{artifact_id}.row_count")
    if row_count != expected_count:
        raise MsrbEvaluationBaselineError(
            f"{artifact_id} row count mismatch: this baseline requires {expected_count}, observed {row_count}."
        )
    return FrozenArtifact(path=relative_path, sha256=expected_sha256, row_count=row_count)


def load_frozen_file(
    value: object,
    *,
    root: Path,
    source_id: str,
) -> FrozenFile:
    """Parse and verify one digest-bound repository file."""

    raw = _mapping(value, context=source_id)
    _exact_fields(raw, {"path", "sha256"}, context=source_id)
    relative_path = _relative_path(raw["path"], context=f"{source_id}.path")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MsrbEvaluationBaselineError(f"{source_id} path escapes the repository.") from exc
    if not path.is_file():
        raise MsrbEvaluationBaselineError(f"{source_id} is missing: {path}")
    expected_sha256 = _sha256_text(raw["sha256"], context=f"{source_id}.sha256")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise MsrbEvaluationBaselineError(
            f"{source_id} SHA-256 mismatch: expected {expected_sha256}, observed {actual_sha256}."
        )
    return FrozenFile(path=relative_path, sha256=expected_sha256)


def verify_baseline_sources(root: Path, baseline: ParsedBaseline) -> SelectionReplayEvidence:
    """Verify alias identity and semantic contents of all frozen Parquet sources."""

    _verify_aliases(root, baseline)
    prediction_frame = _verify_prediction_ledger(root, baseline.prediction_ledger)
    selection_frame, selected_candidate_ids, selected_sequences = _verify_selection_batch(
        root,
        baseline.selection_batch,
        baseline,
    )
    observed_candidate_ids, observed_sequences = _verify_labels_used(
        root,
        baseline.labels_used,
        baseline.comparison_candidate_ids,
    )
    require_selection_disjoint_from_labels(
        selected_candidate_ids=selected_candidate_ids,
        selected_sequences=selected_sequences,
        observed_candidate_ids=observed_candidate_ids,
        observed_sequences=observed_sequences,
    )
    config_path = root / baseline.campaign_config.path
    try:
        campaign_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MsrbEvaluationBaselineError(f"campaign config YAML is invalid: {exc}") from exc
    return verify_msrb_selection_replay(
        prediction_frame=prediction_frame,
        selection_frame=selection_frame,
        campaign_config=campaign_config,
        expected_campaign_slug=CAMPAIGN_SLUG,
        expected_allocation_api_version=baseline.selection_allocation_api_version,
    )


def require_selection_disjoint_from_labels(
    *,
    selected_candidate_ids: set[str],
    selected_sequences: set[str],
    observed_candidate_ids: set[str],
    observed_sequences: set[str],
) -> None:
    """Reject a prospective batch that repeats an already observed identity."""

    candidate_overlap = sorted(selected_candidate_ids.intersection(observed_candidate_ids))
    if candidate_overlap:
        raise MsrbEvaluationBaselineError(
            f"selection_batch and labels_used candidate IDs overlap: {candidate_overlap[:5]}."
        )
    sequence_overlap = sorted(selected_sequences.intersection(observed_sequences))
    if sequence_overlap:
        raise MsrbEvaluationBaselineError(f"selection_batch and labels_used sequences overlap: {sequence_overlap[:5]}.")


def verify_msrb_selection_replay(
    *,
    prediction_frame: pd.DataFrame,
    selection_frame: pd.DataFrame,
    campaign_config: object,
    expected_campaign_slug: str,
    expected_allocation_api_version: str,
) -> SelectionReplayEvidence:
    """Recompute MSRB scores and replay the production sequence-unique allocator."""

    if expected_allocation_api_version != SELECTION_ALLOCATION_PREVIEW_API_VERSION:
        raise MsrbEvaluationBaselineError(
            "selection allocation API version drift: "
            f"expected {expected_allocation_api_version!r}, "
            f"runtime exposes {SELECTION_ALLOCATION_PREVIEW_API_VERSION!r}."
        )
    config = _mapping(campaign_config, context="campaign config")
    campaign = _mapping(config.get("campaign"), context="campaign config.campaign")
    if campaign.get("slug") != expected_campaign_slug:
        raise MsrbEvaluationBaselineError(
            f"campaign config slug drift: expected {expected_campaign_slug!r}, observed {campaign.get('slug')!r}."
        )
    required_prediction_columns = {
        "id",
        "sequence",
        "pred__y_hat_model",
        "pred__selection_views",
    }
    missing_prediction_columns = sorted(required_prediction_columns - set(prediction_frame.columns))
    if missing_prediction_columns:
        raise MsrbEvaluationBaselineError(
            f"prediction ledger is missing selection-replay columns: {missing_prediction_columns}."
        )
    required_selection_columns = {
        "id",
        "selection_batch_key",
        "allocation_view_id",
        "allocation_slot",
    }
    missing_selection_columns = sorted(required_selection_columns - set(selection_frame.columns))
    if missing_selection_columns:
        raise MsrbEvaluationBaselineError(
            f"selection batch is missing allocation-replay columns: {missing_selection_columns}."
        )
    prediction_rows = prediction_frame.loc[:, sorted(required_prediction_columns)].copy()
    _frame_unique(prediction_rows, "id", artifact_id="prediction_ledger")
    _frame_unique(prediction_rows, "sequence", artifact_id="prediction_ledger")
    prediction_rows["id"] = prediction_rows["id"].map(str)
    prediction_rows["sequence"] = prediction_rows["sequence"].map(lambda value: str(value).strip().upper())
    try:
        y_hat = np.stack(
            [np.asarray(value, dtype=float) for value in prediction_rows["pred__y_hat_model"]],
            axis=0,
        )
    except (TypeError, ValueError) as exc:
        raise MsrbEvaluationBaselineError(
            "prediction ledger pred__y_hat_model values must form one finite rectangular matrix."
        ) from exc
    if y_hat.ndim != 2 or not np.isfinite(y_hat).all():
        raise MsrbEvaluationBaselineError(
            "prediction ledger pred__y_hat_model values must form one finite rectangular matrix."
        )

    view_configs = config.get("selection_views")
    if not isinstance(view_configs, list) or not view_configs:
        raise MsrbEvaluationBaselineError("campaign config.selection_views must be a non-empty list.")
    selection_batch_config = _mapping(config.get("selection_batch"), context="campaign config.selection_batch")
    if selection_batch_config.get("deduplicate_by") != "sequence":
        raise MsrbEvaluationBaselineError("campaign selection_batch must deduplicate by sequence.")
    allocation_config = _mapping(
        selection_batch_config.get("allocation"),
        context="campaign config.selection_batch.allocation",
    )
    if allocation_config.get("strategy") != "round_robin_next_best_unallocated":
        raise MsrbEvaluationBaselineError("campaign selection allocation must use round_robin_next_best_unallocated.")
    raw_priority = allocation_config.get("view_priority")
    if (
        not isinstance(raw_priority, list)
        or not raw_priority
        or any(not isinstance(value, str) for value in raw_priority)
    ):
        raise MsrbEvaluationBaselineError("campaign selection view_priority must be a non-empty string list.")
    view_priority = tuple(raw_priority)
    if len(set(view_priority)) != len(view_priority):
        raise MsrbEvaluationBaselineError("campaign selection view_priority must be unique.")

    stored_scores = _stored_selection_scores(
        prediction_rows["pred__selection_views"],
        expected_view_ids=view_priority,
    )
    view_frames: list[pd.DataFrame] = []
    maximum_score_difference = 0.0
    configured_ids: list[str] = []
    expected_unique_count = 0
    for index, raw_view in enumerate(view_configs):
        view = _mapping(raw_view, context=f"campaign config.selection_views[{index}]")
        view_id = _text(view.get("id"), context=f"campaign config.selection_views[{index}].id")
        configured_ids.append(view_id)
        objective = _mapping(
            view.get("objective"),
            context=f"campaign config.selection_views[{index}].objective",
        )
        if objective.get("name") != "multistate_response_behavior_v1":
            raise MsrbEvaluationBaselineError(
                f"campaign selection view {view_id!r} must use multistate_response_behavior_v1."
            )
        objective_params = _mapping(
            objective.get("params"),
            context=f"campaign config.selection_views[{index}].objective.params",
        )
        selection = _mapping(
            view.get("selection"),
            context=f"campaign config.selection_views[{index}].selection",
        )
        if selection.get("name") != "top_n":
            raise MsrbEvaluationBaselineError(f"campaign selection view {view_id!r} must use top_n.")
        selection_params = _mapping(
            selection.get("params"),
            context=f"campaign config.selection_views[{index}].selection.params",
        )
        expected_selection_params = {
            "score_ref": "behavior_score",
            "tie_handling": "ordinal",
            "objective_mode": "maximize",
            "require_exact_top_k": True,
        }
        for field, expected in expected_selection_params.items():
            if selection_params.get(field) != expected:
                raise MsrbEvaluationBaselineError(
                    f"campaign selection view {view_id!r} {field} must equal {expected!r}."
                )
        top_k = _positive_integer(
            selection_params.get("top_k"),
            context=f"campaign selection view {view_id!r} top_k",
        )
        expected_unique_count += top_k
        try:
            recomputed = score_multistate_response_behavior(y_hat, **objective_params).behavior_score
        except Exception as exc:
            raise MsrbEvaluationBaselineError(
                f"could not recompute MSRB scores for selection view {view_id!r}: {exc}"
            ) from exc
        stored = stored_scores[view_id]
        for row_index, entry in enumerate(stored["entries"]):
            if entry.get("objective_name") != "multistate_response_behavior_v1":
                raise MsrbEvaluationBaselineError(f"prediction ledger objective identity drift for view {view_id!r}.")
            if entry.get("selection_name") != "top_n":
                raise MsrbEvaluationBaselineError(f"prediction ledger selection identity drift for view {view_id!r}.")
            if entry.get("score_ref") != f"{view_id}/behavior_score":
                raise MsrbEvaluationBaselineError(f"prediction ledger score_ref drift for view {view_id!r}.")
            if entry.get("top_k") != top_k:
                raise MsrbEvaluationBaselineError(f"prediction ledger top_k drift for view {view_id!r}.")
            try:
                score = float(entry["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MsrbEvaluationBaselineError(
                    f"prediction ledger stored score is invalid for view {view_id!r}, row {row_index}."
                ) from exc
            if not math.isfinite(score):
                raise MsrbEvaluationBaselineError(
                    f"prediction ledger stored score is non-finite for view {view_id!r}, row {row_index}."
                )
        stored_values = stored["scores"]
        differences = np.abs(np.asarray(recomputed, dtype=float) - stored_values)
        view_maximum = float(differences.max(initial=0.0))
        maximum_score_difference = max(maximum_score_difference, view_maximum)
        if view_maximum > 1e-12:
            raise MsrbEvaluationBaselineError(
                f"stored MSRB score drift for view {view_id!r}: maximum absolute difference {view_maximum:.3g}."
            )
        ranked = pd.DataFrame(
            {
                "id": prediction_rows["id"].to_numpy(),
                "score": np.asarray(recomputed, dtype=float),
            }
        ).sort_values(["score", "id"], ascending=[False, True], kind="mergesort")
        ranked = ranked.reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        ranked["selection_view_id"] = view_id
        ranked["top_k"] = top_k
        view_frames.append(ranked)

    if tuple(configured_ids) != view_priority:
        raise MsrbEvaluationBaselineError(
            "campaign selection_views order must exactly match selection_batch allocation view_priority."
        )
    declared_unique_count = _positive_integer(
        selection_batch_config.get("expected_unique_count"),
        context="campaign config.selection_batch.expected_unique_count",
    )
    if declared_unique_count != expected_unique_count:
        raise MsrbEvaluationBaselineError(
            "campaign selection_batch.expected_unique_count must equal the sum of view top_k values."
        )
    try:
        replay = preview_round_robin_next_best_unallocated(
            candidate_rows=prediction_rows.loc[:, ["id", "sequence"]].rename(columns={"sequence": "dedup_key"}),
            view_rows=pd.concat(view_frames, ignore_index=True),
            view_priority=view_priority,
        ).allocated
    except Exception as exc:
        raise MsrbEvaluationBaselineError(f"could not replay sequence-unique selection allocation: {exc}") from exc
    _require_exact_selection_allocation(replay=replay, observed=selection_frame)
    return SelectionReplayEvidence(
        score_count=len(prediction_rows) * len(view_priority),
        max_abs_score_difference=maximum_score_difference,
        allocated_count=len(replay),
    )


def _stored_selection_scores(
    values: pd.Series,
    *,
    expected_view_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    entries_by_view: dict[str, list[dict[str, Any]]] = {view_id: [] for view_id in expected_view_ids}
    scores_by_view: dict[str, list[float]] = {view_id: [] for view_id in expected_view_ids}
    for row_index, raw_entries in enumerate(values.tolist()):
        if not isinstance(raw_entries, (list, tuple, np.ndarray)):
            raise MsrbEvaluationBaselineError(
                f"prediction ledger selection views must be a sequence at row {row_index}."
            )
        by_view: dict[str, dict[str, Any]] = {}
        for raw_entry in raw_entries:
            entry = _mapping(raw_entry, context=f"prediction ledger selection view at row {row_index}")
            view_id = _text(
                entry.get("selection_view_id"),
                context=f"prediction ledger selection view ID at row {row_index}",
            )
            if view_id in by_view:
                raise MsrbEvaluationBaselineError(
                    f"prediction ledger duplicates selection view {view_id!r} at row {row_index}."
                )
            by_view[view_id] = entry
        if set(by_view) != set(expected_view_ids):
            raise MsrbEvaluationBaselineError(
                f"prediction ledger selection views drift at row {row_index}: "
                f"expected {list(expected_view_ids)}, observed {sorted(by_view)}."
            )
        for view_id in expected_view_ids:
            entry = by_view[view_id]
            try:
                score = float(entry["score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MsrbEvaluationBaselineError(
                    f"prediction ledger stored score is invalid for view {view_id!r}, row {row_index}."
                ) from exc
            entries_by_view[view_id].append(entry)
            scores_by_view[view_id].append(score)
    return {
        view_id: {
            "entries": entries_by_view[view_id],
            "scores": np.asarray(scores_by_view[view_id], dtype=float),
        }
        for view_id in expected_view_ids
    }


def _require_exact_selection_allocation(*, replay: pd.DataFrame, observed: pd.DataFrame) -> None:
    columns = {
        "selection_view_id": "allocation_view_id",
        "allocation_slot": "allocation_slot",
        "id": "id",
        "dedup_key": "selection_batch_key",
    }
    expected = replay.loc[:, list(columns)].rename(columns=columns).copy()
    actual = observed.loc[:, list(columns.values())].copy()
    for frame in (expected, actual):
        frame["id"] = frame["id"].map(str)
        frame["selection_batch_key"] = frame["selection_batch_key"].map(lambda value: str(value).strip().upper())
        try:
            frame["allocation_slot"] = frame["allocation_slot"].map(int)
        except (TypeError, ValueError) as exc:
            raise MsrbEvaluationBaselineError("selection allocation slots must be integers.") from exc
    key_columns = ["allocation_view_id", "allocation_slot"]
    if expected.duplicated(key_columns).any() or actual.duplicated(key_columns).any():
        raise MsrbEvaluationBaselineError("selection allocation drift: view and slot keys must be unique.")
    expected = expected.sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    actual = actual.sort_values(key_columns, kind="mergesort").reset_index(drop=True)
    if not expected.equals(actual):
        raise MsrbEvaluationBaselineError("selection allocation drift: recomputed allocation does not match receipt.")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_aliases(root: Path, baseline: ParsedBaseline) -> None:
    registry = load_study_promoter_alias_registry(root, registry_path=baseline.alias_registry_path)
    by_alias = {row.alias: row for row in registry.assignments}
    for allocation in baseline.allocations:
        registered = by_alias.get(allocation.study_alias)
        if registered is None:
            raise MsrbEvaluationBaselineError(f"unknown study alias in allocation: {allocation.study_alias}")
        if (
            registered.candidate_id != allocation.candidate_id
            or registered.sequence_sha256 != allocation.sequence_sha256
        ):
            raise MsrbEvaluationBaselineError(
                f"study alias {allocation.study_alias} does not match its frozen candidate and sequence."
            )
    expected_aliases = tuple(f"SECG-{ordinal:03d}" for ordinal in range(19, 37))
    if tuple(row.study_alias for row in baseline.allocations) != expected_aliases:
        raise MsrbEvaluationBaselineError(
            "round-0 MSRB allocations must be ordered by view and slot with study aliases SECG-019..SECG-036."
        )


def _verify_prediction_ledger(root: Path, artifact: FrozenArtifact) -> pd.DataFrame:
    frame = _read_parquet(
        root / artifact.path,
        artifact_id="prediction_ledger",
        columns=[
            "event",
            "run_id",
            "as_of_round",
            "id",
            "sequence",
            "pred__y_dim",
            "pred__y_hat_model",
            "pred__selection_views",
        ],
    )
    _actual_row_count(frame, artifact, artifact_id="prediction_ledger")
    _single_value(frame, "event", "run_pred", artifact_id="prediction_ledger")
    _single_value(frame, "run_id", RUN_ID, artifact_id="prediction_ledger")
    _single_value(frame, "as_of_round", 0, artifact_id="prediction_ledger")
    _single_value(frame, "pred__y_dim", 8, artifact_id="prediction_ledger")
    _frame_unique(frame, "id", artifact_id="prediction_ledger")
    _frame_unique(frame, "sequence", artifact_id="prediction_ledger")
    return frame


def _verify_selection_batch(
    root: Path,
    artifact: FrozenArtifact,
    baseline: ParsedBaseline,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    frame = _read_parquet(
        root / artifact.path,
        artifact_id="selection_batch",
        columns=[
            "run_id",
            "as_of_round",
            "campaign_slug",
            "id",
            "selection_batch_key",
            "deduplicate_by",
            "allocation_view_id",
            "allocation_slot",
        ],
    )
    _actual_row_count(frame, artifact, artifact_id="selection_batch")
    _single_value(frame, "run_id", RUN_ID, artifact_id="selection_batch")
    _single_value(frame, "as_of_round", 0, artifact_id="selection_batch")
    _single_value(frame, "campaign_slug", CAMPAIGN_SLUG, artifact_id="selection_batch")
    _single_value(frame, "deduplicate_by", "sequence", artifact_id="selection_batch")
    _frame_unique(frame, "id", artifact_id="selection_batch")
    _frame_unique(frame, "selection_batch_key", artifact_id="selection_batch")
    observed_by_id = {str(row.id): row for row in frame.itertuples(index=False)}
    if set(observed_by_id) != {row.candidate_id for row in baseline.allocations}:
        raise MsrbEvaluationBaselineError("selection_batch candidate IDs do not match frozen allocations.")
    for allocation in baseline.allocations:
        observed = observed_by_id[allocation.candidate_id]
        sequence_sha256 = hashlib.sha256(str(observed.selection_batch_key).strip().upper().encode("utf-8")).hexdigest()
        if (
            sequence_sha256 != allocation.sequence_sha256
            or str(observed.allocation_view_id) != allocation.selection_view
            or int(observed.allocation_slot) != allocation.allocation_slot
        ):
            raise MsrbEvaluationBaselineError(
                f"selection_batch allocation drifted for {allocation.study_alias} ({allocation.candidate_id})."
            )
    return (
        frame,
        {str(value) for value in frame["id"].tolist()},
        {str(value).strip().upper() for value in frame["selection_batch_key"].tolist()},
    )


def _verify_labels_used(
    root: Path,
    artifact: FrozenArtifact,
    comparison_ids: tuple[str, ...],
) -> tuple[set[str], set[str]]:
    frame = _read_parquet(
        root / artifact.path,
        artifact_id="labels_used",
        columns=["run_id", "as_of_round", "id", "sequence", "y_obs"],
    )
    _actual_row_count(frame, artifact, artifact_id="labels_used")
    _single_value(frame, "run_id", RUN_ID, artifact_id="labels_used")
    _single_value(frame, "as_of_round", 0, artifact_id="labels_used")
    _frame_unique(frame, "id", artifact_id="labels_used")
    _frame_unique(frame, "sequence", artifact_id="labels_used")
    observed_ids = tuple(sorted(str(value) for value in frame["id"].tolist()))
    if observed_ids != comparison_ids:
        raise MsrbEvaluationBaselineError("comparison candidate IDs do not match labels_used.")
    for value in frame["y_obs"]:
        values = tuple(float(item) for item in value)
        if len(values) != 8 or not all(math.isfinite(item) for item in values):
            raise MsrbEvaluationBaselineError("labels_used y_obs values must be finite eight-component vectors.")
    return (
        {str(value) for value in frame["id"].tolist()},
        {str(value).strip().upper() for value in frame["sequence"].tolist()},
    )


def _read_parquet(path: Path, *, artifact_id: str, columns: list[str]) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        raise MsrbEvaluationBaselineError(f"Could not read required {artifact_id} columns: {exc}") from exc


def _actual_row_count(frame: pd.DataFrame, artifact: FrozenArtifact, *, artifact_id: str) -> None:
    if len(frame) != artifact.row_count:
        raise MsrbEvaluationBaselineError(
            f"{artifact_id} row count mismatch: receipt declares {artifact.row_count}, observed {len(frame)}."
        )


def _single_value(frame: pd.DataFrame, column: str, expected: object, *, artifact_id: str) -> None:
    values = set(frame[column].tolist())
    if values != {expected}:
        raise MsrbEvaluationBaselineError(
            f"{artifact_id} {column} mismatch: expected {expected!r}, observed {sorted(values, key=str)[:5]!r}."
        )


def _frame_unique(frame: pd.DataFrame, column: str, *, artifact_id: str) -> None:
    if frame[column].isna().any() or frame[column].duplicated().any():
        raise MsrbEvaluationBaselineError(f"{artifact_id} {column} values must be non-null and unique.")


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MsrbEvaluationBaselineError(f"{context} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _exact_fields(value: dict[str, Any], expected: set[str], *, context: str) -> None:
    if set(value) != expected:
        raise MsrbEvaluationBaselineError(
            f"{context} fields do not match v1: expected {sorted(expected)}, observed {sorted(value)}."
        )


def _text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MsrbEvaluationBaselineError(f"{context} must be a non-empty string.")
    return value.strip()


def _positive_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MsrbEvaluationBaselineError(f"{context} must be a positive integer.")
    return value


def _sha256_text(value: object, *, context: str) -> str:
    text = _text(value, context=context).lower()
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise MsrbEvaluationBaselineError(f"{context} must be a lowercase SHA-256 digest.")
    return text


def _relative_path(value: object, *, context: str) -> Path:
    path = Path(_text(value, context=context))
    if path.is_absolute() or ".." in path.parts:
        raise MsrbEvaluationBaselineError(f"{context} must be a repository-relative path without '..'.")
    return path
