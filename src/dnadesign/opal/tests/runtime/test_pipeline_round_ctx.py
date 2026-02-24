"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/runtime/test_pipeline_round_ctx.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

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
from dnadesign.opal.src.core.round_context import roundctx_contract
from dnadesign.opal.src.registries.models import list_models, register_model
from dnadesign.opal.src.runtime.run_round import RunRoundRequest, run_round
from dnadesign.opal.src.storage.data_access import RecordsStore
from dnadesign.opal.src.storage.state import CampaignState
from dnadesign.opal.src.transforms_x import identity  # noqa: F401 (registers identity)


def _label_vec(v_logic=1.0, inten=1.0) -> list[float]:
    return [
        0.0,
        0.0,
        0.0,
        float(v_logic),
        float(inten),
        float(inten),
        float(inten),
        float(inten),
    ]


def _ensure_stage_buffer_model_registered() -> None:
    if "test_stage_buffer_model" in list_models():
        return

    @roundctx_contract(
        category="model",
        requires_by_stage={"fit": [], "predict": []},
        produces_by_stage={
            "fit": ["model/<self>/fit_seen"],
            "predict": ["model/<self>/predict_summary"],
        },
    )
    @register_model("test_stage_buffer_model")
    class _Model:
        def __init__(self, params: dict) -> None:
            del params
            self._predict_calls = 0

        def fit(self, X, Y, *, ctx=None):
            del X, Y
            if ctx is not None:
                ctx.set("model/<self>/fit_seen", True)
            return None

        def predict(self, X, *, ctx=None):
            self._predict_calls += 1
            if ctx is not None:
                prev = ctx.get("model/<self>/predict_summary", default={"calls": 0, "rows": 0})
                ctx.set(
                    "model/<self>/predict_summary",
                    {
                        "calls": int(prev["calls"]) + 1,
                        "rows": int(prev["rows"]) + int(X.shape[0]),
                    },
                )
            return np.zeros((X.shape[0], 8), dtype=float)

        def save(self, path: str) -> None:
            Path(path).write_text("ok")

        def get_params(self) -> dict:
            return {}

        @classmethod
        def load(cls, path: str, params: dict | None = None):
            del path
            return cls(params or {})


def test_run_round_writes_round_ctx_and_ledger(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "outputs" / "ledger").mkdir(parents=True, exist_ok=True)
    (workdir / "outputs" / "rounds").mkdir(parents=True, exist_ok=True)
    (workdir / "inputs").mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "sequence": ["AAA", "BBB", "CCC"],
            "bio_type": ["dna", "dna", "dna"],
            "alphabet": ["dna_4", "dna_4", "dna_4"],
            "X": [[0.1], [0.2], [0.3]],
            "opal__demo__label_hist": [
                [{"observed_round": 0, "y_obs": {"value": _label_vec(1.0, 1.0), "dtype": "vector"}}],
                [{"observed_round": 0, "y_obs": {"value": _label_vec(0.8, 2.0), "dtype": "vector"}}],
                [],
            ],
            "Y": [None, None, None],
        }
    )
    df.to_parquet(records_path, index=False)

    cfg = RootConfig(
        campaign=CampaignBlock(name="Demo", slug="demo", workdir=str(workdir)),
        data=DataBlock(
            location=LocationLocal(kind="local", path=str(records_path)),
            x_column_name="X",
            y_column_name="Y",
            transforms_x=PluginRef(name="identity", params={}),
            transforms_y=PluginRef(name="sfxi_vec8_from_table_v1", params={}),
            y_expected_length=8,
        ),
        model=PluginRef(name="random_forest", params={"n_estimators": 5, "random_state": 0}),
        selection=SelectionBlock(
            selection=PluginRef(
                name="top_n",
                params={
                    "top_k": 1,
                    "score_ref": "sfxi_v1/sfxi",
                    "objective_mode": "maximize",
                    "tie_handling": "competition_rank",
                },
            )
        ),
        objectives=ObjectivesBlock(
            objectives=[
                PluginRef(
                    name="sfxi_v1",
                    params={"scaling": {"min_n": 1}},
                )
            ]
        ),
        training=TrainingBlock(policy={"cumulative_training": True}),
        ingest=IngestBlock(duplicate_policy="error"),
        scoring=ScoringBlock(score_batch_size=1000),
        safety=SafetyBlock(
            fail_on_mixed_biotype_or_alphabet=True,
            require_biotype_and_alphabet_on_init=True,
            conflict_policy_on_duplicate_ids="error",
            write_back_requires_columns_present=False,
            accept_x_mismatch=False,
        ),
    )

    # state.json must exist for run_round
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
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )

    tracker = {}

    class _CountingProgress:
        def __init__(self, total: int):
            self.total = int(total)
            self.advanced = 0

        def __enter__(self):
            return self

        def advance(self, n: int = 1) -> None:
            self.advanced += int(n)

        def close(self) -> None:
            return None

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

    def _progress_factory(desc: str, total: int):
        tracker["progress"] = _CountingProgress(total)
        return tracker["progress"]

    res = run_round(
        store,
        store.load(),
        RunRoundRequest(
            cfg=cfg,
            as_of_round=0,
            config_path=workdir / "campaign.yaml",
            progress_factory=_progress_factory,
        ),
    )
    assert res.ok is True
    assert tracker["progress"].total == res.scored
    assert tracker["progress"].advanced == res.scored

    ctx_path = workdir / "outputs" / "rounds" / "round_0" / "metadata" / "round_ctx.json"
    assert ctx_path.exists()
    snap = json.loads(ctx_path.read_text())
    assert "core/data/x_dim" in snap
    assert "transform_x/identity/x_dim" in snap
    produced = snap.get("core/contracts/transform_x/identity/produced", [])
    assert "transform_x/identity/x_dim" in produced
    assert "model/random_forest/fit_metrics" in snap
    produced_model = snap.get("core/contracts/model/random_forest/produced", [])
    assert "model/random_forest/fit_metrics" in produced_model

    assert (workdir / "outputs" / "ledger" / "runs.parquet").exists()
    pred_dir = workdir / "outputs" / "ledger" / "predictions"
    assert pred_dir.exists()
    assert any(p.suffix == ".parquet" for p in pred_dir.iterdir())


def test_run_round_commits_stage_buffered_predict_summary(tmp_path):
    _ensure_stage_buffer_model_registered()
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "outputs" / "ledger").mkdir(parents=True, exist_ok=True)
    (workdir / "outputs" / "rounds").mkdir(parents=True, exist_ok=True)
    (workdir / "inputs").mkdir(parents=True, exist_ok=True)

    records_path = tmp_path / "records.parquet"
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "sequence": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "bio_type": ["dna"] * 6,
            "alphabet": ["dna_4"] * 6,
            "X": [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]],
            "opal__demo__label_hist": [
                [{"observed_round": 0, "y_obs": {"value": _label_vec(1.0, 1.0), "dtype": "vector"}}],
                [{"observed_round": 0, "y_obs": {"value": _label_vec(0.8, 2.0), "dtype": "vector"}}],
                [],
                [],
                [],
                [],
            ],
            "Y": [None, None, None, None, None, None],
        }
    )
    df.to_parquet(records_path, index=False)

    cfg = RootConfig(
        campaign=CampaignBlock(name="Demo", slug="demo", workdir=str(workdir)),
        data=DataBlock(
            location=LocationLocal(kind="local", path=str(records_path)),
            x_column_name="X",
            y_column_name="Y",
            transforms_x=PluginRef(name="identity", params={}),
            transforms_y=PluginRef(name="sfxi_vec8_from_table_v1", params={}),
            y_expected_length=8,
        ),
        model=PluginRef(name="test_stage_buffer_model", params={}),
        selection=SelectionBlock(
            selection=PluginRef(
                name="top_n",
                params={
                    "top_k": 1,
                    "score_ref": "sfxi_v1/sfxi",
                    "objective_mode": "maximize",
                    "tie_handling": "competition_rank",
                },
            )
        ),
        objectives=ObjectivesBlock(
            objectives=[
                PluginRef(
                    name="sfxi_v1",
                    params={"scaling": {"min_n": 1}},
                )
            ]
        ),
        training=TrainingBlock(policy={"cumulative_training": True}),
        ingest=IngestBlock(duplicate_policy="error"),
        scoring=ScoringBlock(score_batch_size=2),
        safety=SafetyBlock(
            fail_on_mixed_biotype_or_alphabet=True,
            require_biotype_and_alphabet_on_init=True,
            conflict_policy_on_duplicate_ids="error",
            write_back_requires_columns_present=False,
            accept_x_mismatch=False,
        ),
    )

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
    (workdir / "campaign.yaml").write_text("campaign: {}")

    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )

    res = run_round(
        store,
        store.load(),
        RunRoundRequest(
            cfg=cfg,
            as_of_round=0,
            config_path=workdir / "campaign.yaml",
        ),
    )
    assert res.ok is True
    expected_batches = int(math.ceil(float(res.scored) / 2.0))

    ctx_path = workdir / "outputs" / "rounds" / "round_0" / "metadata" / "round_ctx.json"
    snap = json.loads(ctx_path.read_text())
    summary = snap.get("model/test_stage_buffer_model/predict_summary")
    assert isinstance(summary, dict)
    assert int(summary["calls"]) == expected_batches
    assert int(summary["rows"]) == int(res.scored)
