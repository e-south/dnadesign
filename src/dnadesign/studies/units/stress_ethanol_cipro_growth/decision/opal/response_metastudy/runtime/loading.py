"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/loading.py

Ledger loading and fail-fast input checks for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.reader_bundle import (
    ReaderResponseBundle,
    load_reader_response_bundle,
)

from ...source_evidence import sfxi_round0_source_evidence_dir
from ..core.contracts import (
    EXPECTED_STRESS_TARGET_VIEW_IDS,
    STRESS_RMF_GREEDY_CAMPAIGN_SLUG,
    STRESS_STATE_IDS,
    MetastudyPaths,
    SfxiEvidenceFrame,
    SfxiSourceProvenance,
    StressCampaignContract,
    StressTargetView,
)
from ..evaluation.model_validation_support import validated_model_params

REQUIRED_PREDICTION_COLUMNS = {
    "id",
    "sequence",
    "pred__y_hat_model",
    "pred__score_selected",
    "sel__rank_competition",
    "sel__is_selected",
    "obj__logic_fidelity",
    "obj__effect_scaled",
}
_VEC8_COLUMNS = ("v00", "v10", "v01", "v11", "y00_star", "y10_star", "y01_star", "y11_star")
_RESPONSE_WINDOW_COLUMNS = ("r00", "r10", "r01", "r11", "b00", "b10", "b01", "b11")


_TARGET_VIEW_LABELS = {
    "ethanol": "Ethanol-responsive",
    "ciprofloxacin": "Ciprofloxacin-responsive",
    "and": "AND-responsive",
}

_RMF_CALIBRATION_FIELDS = {
    "response_separation_min",
    "on_magnitude_min",
    "off_magnitude_max",
    "response_separation_scale",
    "on_magnitude_scale",
    "off_magnitude_scale",
}
_RMF_CALIBRATION_COHORT_FIELDS = {
    "cohort_id",
    "unit",
    "scale_quantile",
    "reader_bundle_manifest_sha256",
    "candidate_bindings_manifest_sha256",
    "unit_count",
    "excluded_nonexact_unit_count",
}


def load_stress_campaign_contract(paths: MetastudyPaths) -> StressCampaignContract:
    """Load the configured stress campaign and derive its typed selection views."""

    campaign_dir = paths.campaign_root / STRESS_RMF_GREEDY_CAMPAIGN_SLUG
    config_path = campaign_dir / "configs/campaign.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Stress campaign config is missing: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stress campaign config must be a mapping: {config_path}")
    campaign = payload.get("campaign")
    if not isinstance(campaign, dict) or campaign.get("slug") != STRESS_RMF_GREEDY_CAMPAIGN_SLUG:
        raise ValueError(f"Stress campaign config must declare slug {STRESS_RMF_GREEDY_CAMPAIGN_SLUG!r}.")
    metadata = campaign.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("study_id") != "stress_ethanol_cipro_growth":
        raise ValueError("Stress campaign metadata must declare the stress study identity.")
    response_reduction_id = metadata.get("response_reduction")
    if not isinstance(response_reduction_id, str) or not response_reduction_id.strip():
        raise ValueError("Stress campaign metadata must declare a response reduction.")
    rmf_calibration_cohort = _parse_rmf_calibration_cohort(metadata.get("rmf_calibration"))
    raw_views = payload.get("selection_views")
    if not isinstance(raw_views, list):
        raise ValueError("Stress campaign must declare selection_views.")
    observed_ids = tuple(str(raw.get("id")) for raw in raw_views if isinstance(raw, dict))
    if observed_ids != EXPECTED_STRESS_TARGET_VIEW_IDS or len(observed_ids) != len(raw_views):
        raise ValueError(
            f"Stress campaign selection views must be exactly {EXPECTED_STRESS_TARGET_VIEW_IDS}; found {observed_ids}."
        )
    target_views = tuple(_parse_target_view(raw) for raw in raw_views)
    rmf_calibration_by_view = {
        target_view.id: _parse_rmf_calibration(raw) for raw, target_view in zip(raw_views, target_views, strict=True)
    }
    if len({view.target_mask for view in target_views}) != len(target_views):
        raise ValueError("Stress campaign selection-view target masks must be unique.")

    data = payload.get("data")
    if not isinstance(data, dict) or int(data.get("y_expected_length", -1)) != 8:
        raise ValueError("Stress campaign must declare one eight-value response label.")
    _validate_campaign_model_io(payload)
    location = data.get("location")
    if not isinstance(location, dict) or location.get("kind") != "usr":
        raise ValueError("Stress campaign must use the study USR candidate table.")
    dataset = location.get("dataset")
    relative_path = location.get("path")
    x_column_name = data.get("x_column_name")
    if not all(isinstance(value, str) and value for value in (dataset, relative_path, x_column_name)):
        raise ValueError("Stress campaign candidate-table identity is incomplete.")
    records_path = (paths.repo_root / "src/dnadesign/usr/datasets" / dataset / "records.parquet").resolve()
    if not records_path.is_file():
        raise FileNotFoundError(f"Stress campaign candidate records are missing: {records_path}")
    model = payload.get("model")
    if not isinstance(model, dict) or model.get("name") != "random_forest" or not isinstance(model.get("params"), dict):
        raise ValueError("Stress campaign must declare random_forest model parameters.")
    validated_model_params(model["params"], preserve_oob_score=True)
    return StressCampaignContract(
        slug=STRESS_RMF_GREEDY_CAMPAIGN_SLUG,
        config_path=config_path.resolve(),
        target_views=target_views,
        candidate_records_path=records_path,
        x_column_name=x_column_name,
        response_reduction_id=response_reduction_id.strip(),
        model_params=dict(model["params"]),
        rmf_calibration_by_view=rmf_calibration_by_view,
        rmf_calibration_cohort=rmf_calibration_cohort,
    )


def assert_campaign_response_reduction(
    campaign: StressCampaignContract,
    *,
    primary_reduction_id: str,
) -> None:
    """Require campaign metadata to bind the exact Reader primary reduction."""

    if campaign.response_reduction_id != str(primary_reduction_id):
        raise ValueError(
            "Stress campaign response reduction disagrees with the Reader primary reduction: "
            f"campaign={campaign.response_reduction_id!r}, Reader={str(primary_reduction_id)!r}."
        )


def load_campaign_reader_bundle(
    paths: MetastudyPaths,
    campaign: StressCampaignContract,
) -> ReaderResponseBundle:
    """Load the configured Reader bundle and verify its campaign reduction."""

    request_path = (
        paths.repo_root
        / "src/dnadesign/studies/units/stress_ethanol_cipro_growth"
        / "response_window_observations/config/reader_response_window.yaml"
    )
    bundle = load_reader_response_bundle(paths.reader_bundle_root, expected_request_path=request_path)
    assert_campaign_response_reduction(campaign, primary_reduction_id=bundle.primary_reduction_id)
    return bundle


def _validate_campaign_model_io(payload: dict[str, object]) -> None:
    transforms_x = payload.get("transforms_x")
    if transforms_x != {"name": "identity", "params": {}}:
        raise ValueError("Stress campaign response screen requires the configured identity X transform.")
    transforms_y = payload.get("transforms_y")
    if not isinstance(transforms_y, dict) or transforms_y.get("name") != "vector_from_table_v1":
        raise ValueError("Stress campaign response screen requires vector_from_table_v1 labels.")
    params = transforms_y.get("params")
    if not isinstance(params, dict):
        raise ValueError("Stress campaign response label transform must declare parameters.")
    if params.get("id_column") != "id" or tuple(params.get("value_columns", ())) != _RESPONSE_WINDOW_COLUMNS:
        raise ValueError(
            "Stress campaign response label transform must preserve the ordered Reader response-window vector "
            f"{_RESPONSE_WINDOW_COLUMNS}."
        )


def _parse_target_view(raw: object) -> StressTargetView:
    if not isinstance(raw, dict):
        raise ValueError("Stress campaign selection-view rows must be mappings.")
    view_id = str(raw.get("id"))
    objective = raw.get("objective")
    if not isinstance(objective, dict) or objective.get("name") != "response_magnitude_feasibility_v1":
        raise ValueError(f"Stress target view {view_id!r} must use response_magnitude_feasibility_v1.")
    params = objective.get("params")
    if not isinstance(params, dict) or tuple(params.get("state_ids", ())) != STRESS_STATE_IDS:
        raise ValueError(f"Stress target view {view_id!r} must use state order {STRESS_STATE_IDS}.")
    raw_mask = params.get("target_mask")
    if not isinstance(raw_mask, list):
        raise ValueError(f"Stress target view {view_id!r} lacks target_mask.")
    return StressTargetView(
        id=view_id,
        label=_TARGET_VIEW_LABELS[view_id],
        target_mask=tuple(float(value) for value in raw_mask),  # type: ignore[arg-type]
    )


def _parse_rmf_calibration(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError("Stress campaign selection-view rows must be mappings.")
    view_id = str(raw.get("id"))
    objective = raw.get("objective")
    params = objective.get("params") if isinstance(objective, dict) else None
    calibration = params.get("calibration") if isinstance(params, dict) else None
    if not isinstance(calibration, dict) or set(calibration) != _RMF_CALIBRATION_FIELDS:
        raise ValueError(
            f"Stress target view {view_id!r} calibration fields must be exactly {sorted(_RMF_CALIBRATION_FIELDS)}."
        )
    parsed = {field: float(calibration[field]) for field in sorted(_RMF_CALIBRATION_FIELDS)}
    if not all(np.isfinite(value) for value in parsed.values()):
        raise ValueError(f"Stress target view {view_id!r} calibration values must be finite.")
    scale_fields = {field for field in _RMF_CALIBRATION_FIELDS if field.endswith("_scale")}
    if any(parsed[field] <= 0.0 for field in scale_fields):
        raise ValueError(f"Stress target view {view_id!r} calibration scales must be positive.")
    return parsed


def _parse_rmf_calibration_cohort(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or set(raw) != _RMF_CALIBRATION_COHORT_FIELDS:
        raise ValueError(
            f"Stress campaign RMF calibration cohort fields must be exactly {sorted(_RMF_CALIBRATION_COHORT_FIELDS)}."
        )
    result = dict(raw)
    if result["cohort_id"] != "exact_primary_reader_candidate_experiments_v1":
        raise ValueError("Stress campaign RMF calibration cohort identity disagrees.")
    if result["unit"] != "reader_candidate_experiment":
        raise ValueError("Stress campaign RMF calibration unit disagrees.")
    if float(result["scale_quantile"]) != 0.9:
        raise ValueError("Stress campaign RMF calibration quantile must be 0.9.")
    for field in ("reader_bundle_manifest_sha256", "candidate_bindings_manifest_sha256"):
        value = str(result[field])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"Stress campaign RMF calibration field {field!r} must be a SHA-256 digest.")
        result[field] = value
    for field in ("unit_count", "excluded_nonexact_unit_count"):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Stress campaign RMF calibration field {field!r} must be a nonnegative integer.")
    if int(result["unit_count"]) == 0:
        raise ValueError("Stress campaign RMF calibration cohort cannot be empty.")
    result["scale_quantile"] = float(result["scale_quantile"])
    return result


def load_sfxi_evidence_frame(
    paths: MetastudyPaths,
    source: SfxiSourceProvenance,
    *,
    target_view: StressTargetView,
    stress_campaign: StressCampaignContract,
) -> SfxiEvidenceFrame:
    """Load immutable SFXI evidence from its persisted source artifacts."""

    if target_view.id != source.target_view_id:
        raise ValueError(
            f"SFXI source provenance {source.source_id!r} targets {source.target_view_id!r}, not {target_view.id!r}."
        )
    campaign_dir = sfxi_round0_source_evidence_dir(
        paths.repo_root,
        source_slug=source.source_campaign_slug,
    )
    predictions_path = campaign_dir / "outputs/ledger/predictions"
    runs_path = campaign_dir / "outputs/ledger/runs.parquet"
    for path in (predictions_path, runs_path):
        if not path.exists():
            raise FileNotFoundError(f"Required SFXI source artifact is missing: {path}")

    predictions = pd.read_parquet(predictions_path)
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"{source.source_id}: prediction ledger missing required columns: {missing}")
    if predictions["id"].duplicated().any():
        raise ValueError(f"{source.source_id}: prediction ledger contains duplicate ids.")
    y_hat = _stack_y_hat(predictions["pred__y_hat_model"], slug=source.source_id)

    runs = pd.read_parquet(runs_path)
    if len(runs) != 1:
        raise ValueError(f"{source.source_id}: expected one run metadata row, found {len(runs)}.")
    run = runs.iloc[0]
    assert_sfxi_run_contract(run, source=source, target_view=target_view)
    denom = float(run["objective__denom_value"])
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(f"{source.source_id}: invalid objective denom {denom}.")
    run_id = str(run["run_id"])
    objective_params = run["objective__params"]
    scaling = objective_params["scaling"]
    model_params = dict(run["model__params"])
    model_params.pop("emit_feature_importance", None)
    y_ops = list(run["training__y_ops"])
    return SfxiEvidenceFrame(
        source=source,
        target_view=target_view,
        predictions=predictions,
        y_hat=y_hat,
        denom=denom,
        run_id=run_id,
        scaling_percentile=int(scaling["percentile"]),
        scaling_min_n=int(scaling["min_n"]),
        scaling_eps=float(scaling["eps"]),
        intensity_log2_offset_delta=float(objective_params["intensity_log2_offset_delta"]),
        records_path=stress_campaign.candidate_records_path,
        x_column_name=stress_campaign.x_column_name,
        model_params=model_params,
        yops_eps=float(y_ops[0]["params"]["eps"]),
        stats_n_train=int(run["stats__n_train"]),
        stats_n_scored=int(run["stats__n_scored"]),
    )


def load_observed_label_ids(paths: MetastudyPaths, source: SfxiSourceProvenance) -> set[str]:
    labels = load_observed_label_frame(paths, source)
    return set(labels["id"].astype(str))


def load_observed_label_frame(paths: MetastudyPaths, source: SfxiSourceProvenance) -> pd.DataFrame:
    source_dir = sfxi_round0_source_evidence_dir(
        paths.repo_root,
        source_slug=source.source_campaign_slug,
    )
    labels_path = source_dir / "outputs/ledger/labels.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Required OPAL label ledger is missing: {labels_path}")
    labels = pd.read_parquet(labels_path)
    missing = {"id", "sequence", "y_obs"} - set(labels.columns)
    if missing:
        raise ValueError(f"{source.source_id}: labels ledger missing required columns: {sorted(missing)}")
    return labels


def load_label_source_frame(
    paths: MetastudyPaths,
    source: SfxiSourceProvenance,
    *,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Load the exact Reader source rows that produced the canonical label ledger."""

    source_id = source.source_id
    source_dir = sfxi_round0_source_evidence_dir(
        paths.repo_root,
        source_slug=source.source_campaign_slug,
    )
    source_path = source_dir / "inputs/r0/reader_vec8_batch0.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Required measured Reader label source is missing: {source_path}")
    source_rows = pd.read_csv(source_path)
    required = {"id", "design_id", "reader_experiment_id", *_VEC8_COLUMNS}
    missing = sorted(required - set(source_rows.columns))
    if missing:
        raise ValueError(f"{source_id}: Reader label source missing required columns: {missing}")
    source_rows["id"] = source_rows["id"].astype(str)
    if source_rows["id"].duplicated().any():
        raise ValueError(f"{source_id}: Reader label source contains duplicate ids.")
    label_ids = labels["id"].astype(str).tolist()
    if set(source_rows["id"]) != set(label_ids):
        missing_ids = sorted(set(label_ids) - set(source_rows["id"]))
        extra_ids = sorted(set(source_rows["id"]) - set(label_ids))
        raise ValueError(
            f"{source_id}: Reader label source identity does not match the label ledger; "
            f"missing={missing_ids[:5]}, extra={extra_ids[:5]}."
        )
    aligned = source_rows.set_index("id").loc[label_ids].reset_index()
    source_y = aligned.loc[:, list(_VEC8_COLUMNS)].to_numpy(dtype=float)
    label_y = _stack_vectors(labels["y_obs"], expected_length=8, field="y_obs")
    if not np.allclose(source_y, label_y, rtol=0.0, atol=1.0e-12):
        max_error = float(np.max(np.abs(source_y - label_y)))
        raise ValueError(f"{source_id}: Reader label source vec8 does not match the ledger; max error={max_error}.")
    return aligned


def assert_candidate_alignment(runs: tuple[SfxiEvidenceFrame, ...]) -> None:
    if not runs:
        raise ValueError("at least one SFXI prediction ledger is required.")
    first = _candidate_identity_sequence_keys(runs[0])
    for run in runs[1:]:
        keys = _candidate_identity_sequence_keys(run)
        if keys != first:
            raise ValueError(
                f"{run.source.source_id}: prediction id-to-sequence mapping is not aligned to "
                f"{runs[0].source.source_id}."
            )


def _candidate_identity_sequence_keys(run: SfxiEvidenceFrame) -> list[tuple[str, str]]:
    required = {"id", "sequence"}
    missing = sorted(required - set(run.predictions.columns))
    if missing:
        raise ValueError(f"{run.source.source_id}: prediction ledger is missing candidate columns {missing}.")
    keys = run.predictions.loc[:, ["id", "sequence"]].astype(str)
    if keys["id"].duplicated().any():
        raise ValueError(f"{run.source.source_id}: prediction ledger contains duplicate candidate ids.")
    return list(keys.itertuples(index=False, name=None))


def assert_shared_observed_labels(label_frames: tuple[pd.DataFrame, ...]) -> None:
    if not label_frames:
        raise ValueError("at least one observed label ledger is required.")
    baseline = _normalized_labels(label_frames[0])
    baseline_y = _stack_vectors(baseline["y_obs"], expected_length=8, field="y_obs")
    for frame in label_frames[1:]:
        candidate = _normalized_labels(frame)
        same_keys = baseline[["id", "sequence"]].equals(candidate[["id", "sequence"]])
        same_y = same_keys and np.array_equal(
            baseline_y,
            _stack_vectors(candidate["y_obs"], expected_length=8, field="y_obs"),
        )
        if not same_y:
            raise ValueError("observed label ledgers are not identical across SFXI source artifacts.")


def assert_sfxi_run_contract(
    run: pd.Series,
    *,
    source: SfxiSourceProvenance,
    target_view: StressTargetView,
) -> None:
    """Validate persisted SFXI metadata against its immutable source contract."""

    slug = source.source_id
    _require_equal(slug, "run id", run["run_id"], source.expected_run_id)
    _require_equal(slug, "objective name", run["objective__name"], "sfxi_v1")
    run_objective = run["objective__params"]
    _require_vector_equal(
        slug,
        "setpoint_vector",
        run_objective["setpoint_vector"],
        target_view.target_mask,
    )
    for key in ("logic_exponent_beta", "intensity_exponent_gamma"):
        _require_equal(slug, f"objective {key}", run_objective[key], 1.0)
    _require_equal(slug, "objective intensity_log2_offset_delta", run_objective["intensity_log2_offset_delta"], 0.0)
    _require_equal(
        slug,
        "persisted denominator percentile",
        run["objective__denom_percentile"],
        run_objective["scaling"]["percentile"],
    )
    persisted_y_ops = list(run["training__y_ops"])
    if len(persisted_y_ops) != 1 or persisted_y_ops[0].get("name") != "intensity_median_iqr":
        raise ValueError(f"{slug}: SFXI run must contain one intensity_median_iqr y-op.")
    _require_equal(slug, "model name", run["model__name"], "random_forest")
    _require_equal(slug, "selection name", run["selection__name"], "top_n")
    selection_params = run["selection__params"]
    _require_equal(slug, "selection top_k", selection_params["top_k"], 6)
    _require_equal(slug, "selection score_ref", run["selection__score_ref"], "sfxi_v1/sfxi")
    _require_equal(slug, "selection objective", run["selection__objective"], "maximize")
    _require_equal(slug, "selection tie handling", run["selection__tie_handling"], "competition_rank")


def _require_vector_equal(slug: str, field: str, actual: object, expected: object) -> None:
    if not np.array_equal(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)):
        raise ValueError(f"{slug}: persisted {field} does not match its derived target view.")


def _require_equal(slug: str, field: str, actual: object, expected: object) -> None:
    try:
        numeric_equal = bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=1.0e-12))
    except (TypeError, ValueError):
        numeric_equal = False
    if not numeric_equal and actual != expected:
        raise ValueError(f"{slug}: persisted {field} {actual!r} does not match the SFXI source contract {expected!r}.")


def load_training_matrix(
    records_path: Path,
    *,
    x_column: str,
    labels: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if labels["id"].astype(str).duplicated().any():
        raise ValueError("observed labels contain duplicate ids.")
    label_ids = labels["id"].astype(str).tolist()
    x = load_candidate_matrix(records_path, x_column=x_column, candidate_ids=label_ids)
    y = _stack_vectors(labels["y_obs"], expected_length=8, field="y_obs")
    return x, y


def load_candidate_matrix(
    records_path: Path,
    *,
    x_column: str,
    candidate_ids: list[str],
) -> np.ndarray:
    """Load one candidate X row per exact ID in the requested order."""

    ids = [str(value) for value in candidate_ids]
    if not ids or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("candidate matrix ids must be non-empty and unique.")
    records = pd.read_parquet(
        records_path,
        columns=["id", x_column],
        filters=[("id", "in", ids)],
    )
    records["id"] = records["id"].astype(str)
    if records["id"].duplicated().any():
        raise ValueError("candidate records contain duplicate requested ids.")
    missing = sorted(set(ids) - set(records["id"]))
    if missing:
        raise ValueError(f"candidate records are missing requested ids: {missing[:5]}")
    aligned = records.set_index("id").loc[ids]
    return _stack_vectors(aligned[x_column], expected_length=None, field=x_column)


def _stack_y_hat(values: pd.Series, *, slug: str) -> np.ndarray:
    rows = [np.asarray(value, dtype=float).ravel() for value in values]
    if not rows:
        raise ValueError(f"{slug}: empty prediction ledger.")
    lengths = {row.size for row in rows}
    if lengths != {8}:
        raise ValueError(f"{slug}: expected pred__y_hat_model length 8 for every row; found {sorted(lengths)}.")
    y_hat = np.vstack(rows)
    if not np.all(np.isfinite(y_hat)):
        raise ValueError(f"{slug}: pred__y_hat_model contains non-finite values.")
    return y_hat


def _stack_vectors(
    values: pd.Series,
    *,
    expected_length: int | None,
    field: str,
) -> np.ndarray:
    rows = [np.asarray(value, dtype=float).ravel() for value in values]
    lengths = {row.size for row in rows}
    if not rows or len(lengths) != 1:
        raise ValueError(f"{field} vectors must be non-empty and have one fixed length; found {sorted(lengths)}.")
    if expected_length is not None and lengths != {expected_length}:
        raise ValueError(f"{field} vectors must have length {expected_length}; found {sorted(lengths)}.")
    matrix = np.vstack(rows)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field} vectors must be finite.")
    return matrix


def _normalized_labels(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "sequence", "y_obs"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"observed label ledger missing required columns: {missing}")
    normalized = frame[["id", "sequence", "y_obs"]].copy()
    normalized["id"] = normalized["id"].astype(str)
    normalized["sequence"] = normalized["sequence"].astype(str)
    if normalized["id"].duplicated().any():
        raise ValueError("observed label ledger contains duplicate ids.")
    return normalized.sort_values("id", kind="mergesort").reset_index(drop=True)
