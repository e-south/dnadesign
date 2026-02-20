"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_run_round_integrity.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from dnadesign.opal.src.config.types import (
    CampaignBlock,
    DataBlock,
    IngestBlock,
    LocationLocal,
    ObjectivesBlock,
    PluginRef,
    RootConfig,
    SafetyBlock,
    ScoringBlock,
    SelectionBlock,
    TrainingBlock,
)
from dnadesign.opal.src.core.objective_result import ObjectiveResultV2
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.registries.models import list_models, register_model
from dnadesign.opal.src.registries.objectives import list_objectives, register_objective
from dnadesign.opal.src.registries.selection import list_selections, register_selection
from dnadesign.opal.src.registries.transforms_y import list_y_ops, register_y_op
from dnadesign.opal.src.runtime.round import writebacks as round_writebacks
from dnadesign.opal.src.runtime.round.stages import _format_summary_stats_for_log
from dnadesign.opal.src.runtime.run_round import RunRoundRequest, run_round
from dnadesign.opal.src.storage.data_access import RecordsStore
from dnadesign.opal.src.storage.state import CampaignState, RoundEntry
from dnadesign.opal.src.transforms_x import identity  # noqa: F401 (registers identity)


def _register_bad_length_objective() -> str:
    name = "test_bad_length_v1"
    if name in list_objectives():
        return name

    @register_objective(name)
    def _bad_length_objective(
        *, y_pred, params: Dict[str, Any], ctx=None, train_view=None, y_pred_std=None
    ) -> ObjectiveResultV2:
        _ = y_pred, params, ctx, train_view, y_pred_std
        return ObjectiveResultV2(scores_by_name={"scalar": np.asarray([0.1], dtype=float)})

    return name


def _register_scalar_uq_objective() -> str:
    name = "test_scalar_uq_v1"
    if name in list_objectives():
        return name

    @register_objective(name)
    def _scalar_uq_objective(
        *, y_pred, params: Dict[str, Any], ctx=None, train_view=None, y_pred_std=None
    ) -> ObjectiveResultV2:
        del params, ctx, train_view
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        if y_pred_std is None:
            raise ValueError("y_pred_std is required for test_scalar_uq_v1")
        ys = np.asarray(y_pred_std, dtype=float).reshape(-1)
        if ys.size != yp.size:
            raise ValueError("y_pred_std length mismatch")
        return ObjectiveResultV2(
            scores_by_name={"scalar": yp},
            uncertainty_by_name={"scalar_var": ys**2},
            diagnostics={},
            modes_by_name={"scalar": "maximize"},
        )

    return name


def _register_scalar_non_positive_uq_objective() -> str:
    name = "test_scalar_non_positive_uq_v1"
    if name in list_objectives():
        return name

    @register_objective(name)
    def _scalar_non_positive_uq_objective(
        *, y_pred, params: Dict[str, Any], ctx=None, train_view=None, y_pred_std=None
    ) -> ObjectiveResultV2:
        del params, ctx, train_view, y_pred_std
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        sigma = np.zeros_like(yp)
        return ObjectiveResultV2(
            scores_by_name={"scalar": yp},
            uncertainty_by_name={"scalar_var": sigma},
            diagnostics={},
            modes_by_name={"scalar": "maximize"},
        )

    return name


def _register_non_tie_scalar_objective() -> str:
    name = "test_non_tie_scalar_v1"
    if name in list_objectives():
        return name

    @register_objective(name)
    def _objective(*, y_pred, params: Dict[str, Any], ctx=None, train_view=None, y_pred_std=None) -> ObjectiveResultV2:
        del y_pred, params, ctx, train_view, y_pred_std
        # Deterministic non-tie objective channel to expose ranking/tie handling bugs.
        scores = np.asarray([0.9, 0.8], dtype=float)
        return ObjectiveResultV2(scores_by_name={"scalar": scores}, modes_by_name={"scalar": "maximize"})

    return name


def _register_fixed_tie_selection() -> str:
    name = "test_fixed_tie_selection_v1"
    if name in list_selections():
        return name

    @register_selection(name)
    def _selection(
        *,
        ids,
        scores,
        top_k,
        objective,
        tie_handling,
        ctx=None,
        scalar_uncertainty=None,
        **_,
    ):
        del scores, top_k, objective, tie_handling, ctx, scalar_uncertainty
        n = len(np.asarray(ids).reshape(-1))
        if n != 2:
            raise ValueError("test_fixed_tie_selection_v1 expects exactly two candidates.")
        # Plugin score ties both candidates at the top; competition rank with top_k=1 should select both.
        return {"order_idx": np.array([0, 1], dtype=int), "score": np.array([1.0, 1.0], dtype=float)}

    return name


class _NoopYOpParams(BaseModel):
    offset: float = 1.0


def _register_test_add1_yop_without_inverse_std() -> str:
    name = "test_add1_no_inverse_std"
    if name in list_y_ops():
        return name

    @register_y_op(name)
    def _factory():
        def fit(Y, params, ctx=None):
            _ = Y, params, ctx
            return None

        def transform(Y, params, ctx=None):
            _ = ctx
            return np.asarray(Y, dtype=float) + float(params.offset)

        def inverse(Y, params, ctx=None):
            _ = ctx
            return np.asarray(Y, dtype=float) - float(params.offset)

        return fit, transform, inverse, _NoopYOpParams

    return name


def _register_model_artifacts_runtime_error_model() -> str:
    name = "test_model_artifacts_runtime_error_v1"
    if name in list_models():
        return name

    @register_model(name)
    class _Model:
        def __init__(self, params: Dict[str, Any]) -> None:
            self.params = dict(params or {})
            self._y_dim = 1

        def get_params(self) -> Dict[str, Any]:
            return dict(self.params)

        def fit(self, X, Y, *, ctx=None):
            del X, ctx
            self._y_dim = int(np.asarray(Y, dtype=float).shape[1])
            return {"ok": True}

        def predict(self, X, *, ctx=None):
            del ctx
            X = np.asarray(X, dtype=float)
            return np.zeros((X.shape[0], self._y_dim), dtype=float)

        def save(self, path: str) -> None:
            Path(path).write_bytes(b"model")

        def model_artifacts(self):
            raise RuntimeError("model_artifacts runtime failure")

    return name


def test_format_summary_stats_for_log_handles_mixed_types() -> None:
    summary_stats = {
        "score_max": 1.23456789,
        "uncertainty_method": "delta",
        "train_effect_pool_size": 4,
        "non_finite_value": np.nan,
    }
    kvs = _format_summary_stats_for_log(summary_stats)
    assert kvs == [
        "non_finite_value=nan",
        "score_max=1.23457",
        "train_effect_pool_size=4",
        "uncertainty_method=delta",
    ]


def _write_state(workdir: Path, cfg: RootConfig, records_path: Path) -> None:
    st = CampaignState(
        campaign_slug=cfg.campaign.slug,
        campaign_name=cfg.campaign.name,
        workdir=str(workdir.resolve()),
        data_location={
            "kind": "local",
            "path": str(Path(cfg.data.location.path).resolve()),
            "records_path": str(records_path.resolve()),
        },
        x_column_name=cfg.data.x_column_name,
        y_column_name=cfg.data.y_column_name,
    )
    st.save(workdir / "state.json")


def _make_config(
    *,
    workdir: Path,
    records_path: Path,
    objective_name: str,
    objective_params: Optional[Dict[str, Any]] = None,
    model_name: str = "random_forest",
    model_params: Optional[Dict[str, Any]] = None,
    selection_name: str = "top_n",
    selection_params: Optional[Dict[str, Any]] = None,
    y_expected_length: int = 1,
    score_batch_size: int = 1000,
) -> RootConfig:
    resolved_sel = dict(selection_params or {"top_k": 1})
    resolved_sel.setdefault("score_ref", f"{objective_name}/scalar")
    resolved_sel.setdefault("tie_handling", "competition_rank")
    resolved_sel.setdefault("objective_mode", "maximize")
    if model_params is None:
        model_params = {"n_estimators": 5, "random_state": 0}
    return RootConfig(
        campaign=CampaignBlock(name="Demo", slug="demo", workdir=str(workdir)),
        data=DataBlock(
            location=LocationLocal(kind="local", path=str(records_path)),
            x_column_name="X",
            y_column_name="Y",
            transforms_x=PluginRef(name="identity", params={}),
            transforms_y=PluginRef(name="scalar_from_table_v1", params={}),
            y_expected_length=y_expected_length,
        ),
        model=PluginRef(name=model_name, params=model_params),
        selection=SelectionBlock(selection=PluginRef(name=selection_name, params=resolved_sel)),
        objectives=ObjectivesBlock(objectives=[PluginRef(name=objective_name, params=objective_params or {})]),
        training=TrainingBlock(policy={"cumulative_training": True}),
        ingest=IngestBlock(duplicate_policy="error"),
        scoring=ScoringBlock(score_batch_size=int(score_batch_size)),
        safety=SafetyBlock(
            fail_on_mixed_biotype_or_alphabet=True,
            require_biotype_and_alphabet_on_init=True,
            conflict_policy_on_duplicate_ids="error",
            write_back_requires_columns_present=False,
            accept_x_mismatch=False,
        ),
    )


def _write_records(records_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", "BBB"],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[0.1], [0.2]],
            "opal__demo__label_hist": [
                [
                    {
                        "kind": "label",
                        "observed_round": 0,
                        "y_obs": {"value": [1.0], "dtype": "vector", "schema": {"length": 1}},
                    }
                ],
                [],
            ],
            "Y": [None, None],
        }
    )
    df.to_parquet(records_path, index=False)
    return df


def _write_records_missing_sequence(records_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", None],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[0.1], [0.2]],
            "opal__demo__label_hist": [
                [
                    {
                        "kind": "label",
                        "observed_round": 0,
                        "y_obs": {"value": [1.0], "dtype": "vector", "schema": {"length": 1}},
                    }
                ],
                [],
            ],
            "Y": [None, None],
        }
    )
    df.to_parquet(records_path, index=False)
    return df


def _write_records_scalar_multibatch(records_path: Path) -> pd.DataFrame:
    ids = ["a", "b", "c", "d", "e", "f"]
    df = pd.DataFrame(
        {
            "id": ids,
            "sequence": [f"{i}{i}{i}" for i in ids],
            "bio_type": ["dna"] * len(ids),
            "alphabet": ["dna_4"] * len(ids),
            "X": [[float(ix) / 10.0] for ix in range(len(ids))],
            "opal__demo__label_hist": [
                [
                    {
                        "kind": "label",
                        "observed_round": 0,
                        "y_obs": {"value": [1.0], "dtype": "vector", "schema": {"length": 1}},
                    }
                ],
                [],
                [],
                [],
                [],
                [],
            ],
            "Y": [None] * len(ids),
        }
    )
    df.to_parquet(records_path, index=False)
    return df


def _write_records_vec8(records_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "sequence": ["AAA", "BBB", "CCC"],
            "bio_type": ["dna", "dna", "dna"],
            "alphabet": ["dna_4", "dna_4", "dna_4"],
            "X": [[0.1], [0.2], [0.3]],
            "opal__demo__label_hist": [
                [
                    {
                        "kind": "label",
                        "observed_round": 0,
                        "y_obs": {
                            "value": [0.0, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 0.8],
                            "dtype": "vector",
                            "schema": {"length": 8},
                        },
                    }
                ],
                [],
                [],
            ],
            "Y": [None, None, None],
        }
    )
    df.to_parquet(records_path, index=False)
    return df


def test_run_round_rejects_objective_score_length_mismatch(tmp_path: Path) -> None:
    objective_name = _register_bad_length_objective()
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name=objective_name,
        selection_params={"top_k": 1, "exclude_already_labeled": False},
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    with pytest.raises(OpalError, match="score channel 'scalar' has length"):
        run_round(
            store,
            store.load(),
            RunRoundRequest(
                cfg=cfg,
                as_of_round=0,
                config_path=workdir / "campaign.yaml",
                verbose=False,
            ),
        )


def test_run_round_blocks_when_round_dir_has_any_entry_without_resume(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="scalar_identity_v1",
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    round_dir = workdir / "outputs" / "rounds" / "round_0"
    round_dir.mkdir(parents=True, exist_ok=True)
    round_log = round_dir / "logs" / "round.log.jsonl"
    round_log.parent.mkdir(parents=True, exist_ok=True)
    round_log.write_text("stale\n")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    with pytest.raises(OpalError, match="already contains"):
        run_round(
            store,
            store.load(),
            RunRoundRequest(
                cfg=cfg,
                as_of_round=0,
                config_path=workdir / "campaign.yaml",
                verbose=False,
            ),
        )


def test_run_round_resume_cleans_round_dir(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="scalar_identity_v1",
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    round_dir = workdir / "outputs" / "rounds" / "round_0"
    stale_dir = round_dir / "stale_dir"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale_file = round_dir / "stale.txt"
    stale_file.write_text("stale\n")
    (stale_dir / "nested.txt").write_text("nested\n")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    res = run_round(
        store,
        store.load(),
        RunRoundRequest(
            cfg=cfg,
            as_of_round=0,
            config_path=workdir / "campaign.yaml",
            verbose=False,
            allow_resume=True,
        ),
    )
    assert res.ok is True
    assert not stale_file.exists()
    assert not stale_dir.exists()


def test_run_round_allow_resume_rejects_malformed_state_round_index(tmp_path: Path, monkeypatch) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="scalar_identity_v1",
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    bad_state = CampaignState(
        campaign_slug=cfg.campaign.slug,
        campaign_name=cfg.campaign.name,
        workdir=str(workdir.resolve()),
        data_location={"kind": "local", "path": str(records_path.resolve())},
        x_column_name=cfg.data.x_column_name,
        y_column_name=cfg.data.y_column_name,
    )
    bad_state.rounds = [
        RoundEntry(
            round_index="not-an-int",  # type: ignore[arg-type]
            run_id="run-prev",
            round_name="round_prev",
            round_dir=str(workdir / "outputs" / "rounds" / "round_prev"),
            labels_used_rounds=[0],
            number_of_training_examples_used_in_round=1,
            number_of_candidates_scored_in_round=1,
            selection_top_k_requested=1,
            selection_top_k_effective_after_ties=1,
            model={},
            metrics={},
            durations_sec={},
            seeds={},
            artifacts={},
            writebacks={},
            warnings=[],
        )
    ]
    monkeypatch.setattr(
        round_writebacks.CampaignState,
        "load",
        classmethod(lambda cls, path: bad_state),
    )

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    with pytest.raises(ValueError, match="invalid literal"):
        run_round(
            store,
            store.load(),
            RunRoundRequest(
                cfg=cfg,
                as_of_round=0,
                config_path=workdir / "campaign.yaml",
                verbose=False,
                allow_resume=True,
            ),
        )


def test_run_round_preserves_null_sequence_in_selection_artifacts(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records_missing_sequence(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="scalar_identity_v1",
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    res = run_round(
        store,
        store.load(),
        RunRoundRequest(
            cfg=cfg,
            as_of_round=0,
            config_path=workdir / "campaign.yaml",
            verbose=False,
        ),
    )
    assert res.ok is True

    sel_csv_path = workdir / "outputs" / "rounds" / "round_0" / "selection" / "selection_top_k.csv"
    sel_parquet_path = sel_csv_path.with_suffix(".parquet")
    assert sel_csv_path.exists()
    assert not sel_parquet_path.exists()
    sel_df = pd.read_csv(sel_csv_path)
    assert pd.isna(sel_df.loc[0, "sequence"])


def test_run_round_bubbles_runtime_error_from_model_artifacts(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    model_name = _register_model_artifacts_runtime_error_model()
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="scalar_identity_v1",
        model_name=model_name,
        model_params={},
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    with pytest.raises(RuntimeError, match=r"^model_artifacts runtime failure$"):
        run_round(
            store,
            store.load(),
            RunRoundRequest(
                cfg=cfg,
                as_of_round=0,
                config_path=workdir / "campaign.yaml",
                verbose=False,
            ),
        )


def test_run_round_matrix_gp_top_n_path(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="scalar_identity_v1",
        model_name="gaussian_process",
        model_params={
            "normalize_y": False,
            "alpha": 1e-6,
            "kernel": {"name": "matern", "length_scale": 0.5, "nu": 1.5, "with_white_noise": True},
        },
        selection_params={"top_k": 1, "score_ref": "scalar_identity_v1/scalar"},
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    res = run_round(
        store,
        store.load(),
        RunRoundRequest(
            cfg=cfg,
            as_of_round=0,
            config_path=workdir / "campaign.yaml",
            verbose=False,
        ),
    )
    assert res.ok is True


def test_run_round_matrix_rf_sfxi_top_n_path(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records_vec8(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="sfxi_v1",
        objective_params={"setpoint_vector": [0, 0, 0, 1], "scaling": {"min_n": 1}},
        selection_params={"top_k": 1, "score_ref": "sfxi_v1/sfxi"},
        y_expected_length=8,
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    res = run_round(
        store,
        store.load(),
        RunRoundRequest(
            cfg=cfg,
            as_of_round=0,
            config_path=workdir / "campaign.yaml",
            verbose=False,
        ),
    )
    assert res.ok is True


def test_run_round_matrix_gp_ei_path(tmp_path: Path) -> None:
    objective_name = _register_scalar_uq_objective()
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name=objective_name,
        model_name="gaussian_process",
        model_params={"normalize_y": False, "alpha": 1e-6},
        selection_name="expected_improvement",
        selection_params={
            "top_k": 1,
            "score_ref": f"{objective_name}/scalar",
            "uncertainty_ref": f"{objective_name}/scalar_var",
            "objective_mode": "maximize",
            "alpha": 1.0,
            "beta": 1.0,
        },
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    res = run_round(
        store,
        store.load(),
        RunRoundRequest(
            cfg=cfg,
            as_of_round=0,
            config_path=workdir / "campaign.yaml",
            verbose=False,
        ),
    )
    assert res.ok is True
    df_after = store.load()
    hist = df_after.loc[df_after["id"] == "b", "opal__demo__label_hist"].iloc[0]
    pred_entries = [e for e in hist if e.get("kind") == "pred"]
    assert pred_entries
    metrics = pred_entries[-1]["metrics"]
    assert "score" in metrics
    assert np.isfinite(float(metrics["score"]))
    assert 0.0 <= float(metrics["score"]) <= 1.0
    assert f"score::{objective_name}/scalar" in metrics
    assert f"uncertainty::{objective_name}/scalar_var" in metrics


def test_run_round_ei_fails_on_non_positive_uncertainty_channel(tmp_path: Path) -> None:
    objective_name = _register_scalar_non_positive_uq_objective()
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name=objective_name,
        model_name="random_forest",
        model_params={"n_estimators": 5, "random_state": 0},
        selection_name="expected_improvement",
        selection_params={
            "top_k": 1,
            "score_ref": f"{objective_name}/scalar",
            "uncertainty_ref": f"{objective_name}/scalar_var",
            "objective_mode": "maximize",
            "alpha": 1.0,
            "beta": 1.0,
        },
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    with pytest.raises(OpalError, match="must be > 0"):
        run_round(
            store,
            store.load(),
            RunRoundRequest(
                cfg=cfg,
                as_of_round=0,
                config_path=workdir / "campaign.yaml",
                verbose=False,
            ),
        )


def test_run_round_rejects_ei_when_yops_lacks_inverse_std(tmp_path: Path) -> None:
    objective_name = _register_scalar_uq_objective()
    yop_name = _register_test_add1_yop_without_inverse_std()
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name=objective_name,
        model_name="gaussian_process",
        model_params={"normalize_y": False, "alpha": 1e-6},
        selection_name="expected_improvement",
        selection_params={
            "top_k": 1,
            "score_ref": f"{objective_name}/scalar",
            "uncertainty_ref": f"{objective_name}/scalar_var",
            "objective_mode": "maximize",
            "alpha": 1.0,
            "beta": 1.0,
        },
    )
    cfg.training.y_ops = [PluginRef(name=yop_name, params={"offset": 1.0})]
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    with pytest.raises(OpalError, match="inverse_std"):
        run_round(
            store,
            store.load(),
            RunRoundRequest(
                cfg=cfg,
                as_of_round=0,
                config_path=workdir / "campaign.yaml",
                verbose=False,
            ),
        )


def test_run_round_rejects_topn_when_yops_lacks_inverse_std_with_gp(tmp_path: Path) -> None:
    yop_name = _register_test_add1_yop_without_inverse_std()
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="scalar_identity_v1",
        model_name="gaussian_process",
        model_params={"normalize_y": False, "alpha": 1e-6},
        selection_name="top_n",
        selection_params={
            "top_k": 1,
            "score_ref": "scalar_identity_v1/scalar",
            "objective_mode": "maximize",
        },
    )
    cfg.training.y_ops = [PluginRef(name=yop_name, params={"offset": 1.0})]
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    with pytest.raises(OpalError, match="inverse_std"):
        run_round(
            store,
            store.load(),
            RunRoundRequest(
                cfg=cfg,
                as_of_round=0,
                config_path=workdir / "campaign.yaml",
                verbose=False,
            ),
        )


def test_run_round_gp_ei_with_intensity_yops_emits_finite_uncertainty(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records_vec8(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="sfxi_v1",
        objective_params={"setpoint_vector": [0, 0, 0, 1], "scaling": {"min_n": 1}},
        model_name="gaussian_process",
        model_params={
            "normalize_y": False,
            "alpha": 1e-6,
            "kernel": {"name": "matern", "length_scale": 0.5, "nu": 1.5, "with_white_noise": True},
        },
        selection_name="expected_improvement",
        selection_params={
            "top_k": 1,
            "score_ref": "sfxi_v1/sfxi",
            "uncertainty_ref": "sfxi_v1/sfxi",
            "objective_mode": "maximize",
            "alpha": 1.0,
            "beta": 1.0,
        },
        y_expected_length=8,
    )
    cfg.training.y_ops = [PluginRef(name="intensity_median_iqr", params={"min_labels": 1})]
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    res = run_round(
        store,
        store.load(),
        RunRoundRequest(
            cfg=cfg,
            as_of_round=0,
            config_path=workdir / "campaign.yaml",
            verbose=False,
        ),
    )
    assert res.ok is True

    df_after = store.load()
    emitted = []
    for hist in df_after["opal__demo__label_hist"]:
        for event in hist:
            if event.get("kind") != "pred":
                continue
            metrics = event.get("metrics", {}) or {}
            if "uncertainty::sfxi_v1/sfxi" in metrics:
                emitted.append(float(metrics["uncertainty::sfxi_v1/sfxi"]))
    assert emitted, "Expected uncertainty::sfxi_v1/sfxi to be emitted in prediction metrics."
    assert np.all(np.isfinite(np.asarray(emitted, dtype=float)))
    assert np.all(np.asarray(emitted, dtype=float) >= 0.0)


def test_run_round_matrix_gp_topn_handles_scalar_uncertainty_multibatch(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records_scalar_multibatch(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="scalar_identity_v1",
        model_name="gaussian_process",
        model_params={"normalize_y": False, "alpha": 1e-6},
        selection_name="top_n",
        selection_params={"top_k": 2, "score_ref": "scalar_identity_v1/scalar", "objective_mode": "maximize"},
        score_batch_size=2,
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    res = run_round(
        store,
        store.load(),
        RunRoundRequest(
            cfg=cfg,
            as_of_round=0,
            config_path=workdir / "campaign.yaml",
            verbose=False,
        ),
    )
    assert res.ok is True


def test_run_round_uses_selection_score_for_tie_expansion(tmp_path: Path) -> None:
    objective_name = _register_non_tie_scalar_objective()
    selection_name = _register_fixed_tie_selection()

    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name=objective_name,
        selection_name=selection_name,
        selection_params={
            "top_k": 1,
            "score_ref": f"{objective_name}/scalar",
            "objective_mode": "maximize",
            "tie_handling": "competition_rank",
            "exclude_already_labeled": False,
        },
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    res = run_round(
        store,
        store.load(),
        RunRoundRequest(
            cfg=cfg,
            as_of_round=0,
            config_path=workdir / "campaign.yaml",
            verbose=False,
        ),
    )
    assert res.ok is True

    sel_csv_path = workdir / "outputs" / "rounds" / "round_0" / "selection" / "selection_top_k.csv"
    sel_df = pd.read_csv(sel_csv_path)
    assert sel_df.shape[0] == 2


def test_run_round_rejects_invalid_score_ref(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="scalar_identity_v1",
        selection_params={"top_k": 1, "score_ref": "scalar_identity_v1/missing"},
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    with pytest.raises(OpalError, match="score_ref channel"):
        run_round(
            store,
            store.load(),
            RunRoundRequest(
                cfg=cfg,
                as_of_round=0,
                config_path=workdir / "campaign.yaml",
                verbose=False,
            ),
        )


def test_run_round_rejects_missing_objective_mode(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="scalar_identity_v1",
        selection_params={"top_k": 1, "score_ref": "scalar_identity_v1/scalar"},
    )
    cfg.selection.selection.params.pop("objective_mode", None)
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    with pytest.raises(OpalError, match="objective_mode"):
        run_round(
            store,
            store.load(),
            RunRoundRequest(
                cfg=cfg,
                as_of_round=0,
                config_path=workdir / "campaign.yaml",
                verbose=False,
            ),
        )


def test_run_round_rejects_missing_tie_handling(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name="scalar_identity_v1",
        selection_params={"top_k": 1, "score_ref": "scalar_identity_v1/scalar", "objective_mode": "maximize"},
    )
    cfg.selection.selection.params.pop("tie_handling", None)
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    with pytest.raises(OpalError, match="tie_handling"):
        run_round(
            store,
            store.load(),
            RunRoundRequest(
                cfg=cfg,
                as_of_round=0,
                config_path=workdir / "campaign.yaml",
                verbose=False,
            ),
        )


def test_run_round_rejects_invalid_uncertainty_ref(tmp_path: Path) -> None:
    objective_name = _register_scalar_uq_objective()
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    _write_records(records_path)
    cfg = _make_config(
        workdir=workdir,
        records_path=records_path,
        objective_name=objective_name,
        model_name="gaussian_process",
        model_params={"normalize_y": False, "alpha": 1e-6},
        selection_name="expected_improvement",
        selection_params={
            "top_k": 1,
            "score_ref": f"{objective_name}/scalar",
            "uncertainty_ref": f"{objective_name}/does_not_exist",
            "objective_mode": "maximize",
        },
    )
    _write_state(workdir, cfg, records_path)
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params={},
    )

    with pytest.raises(OpalError, match="uncertainty_ref channel"):
        run_round(
            store,
            store.load(),
            RunRoundRequest(
                cfg=cfg,
                as_of_round=0,
                config_path=workdir / "campaign.yaml",
                verbose=False,
            ),
        )
