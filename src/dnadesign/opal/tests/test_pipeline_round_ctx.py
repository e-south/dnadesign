"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_pipeline_round_ctx.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
from pathlib import Path

import pandas as pd

from dnadesign.opal.src.config.types import (
    CampaignBlock,
    DataBlock,
    IngestBlock,
    LocationLocal,
    MetadataBlock,
    ObjectiveBlock,
    PluginRef,
    RootConfig,
    SafetyBlock,
    ScoringBlock,
    SelectionBlock,
    TrainingBlock,
)
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


def test_run_round_writes_round_ctx_and_ledger(tmp_path):
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "outputs").mkdir(parents=True, exist_ok=True)
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
                [{"r": 0, "y": _label_vec(1.0, 1.0)}],
                [{"r": 0, "y": _label_vec(0.8, 2.0)}],
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
        selection=SelectionBlock(selection=PluginRef(name="top_n", params={"top_k": 1})),
        objective=ObjectiveBlock(
            objective=PluginRef(
                name="sfxi_v1",
                params={"scaling": {"min_n": 1}},
            )
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
        metadata=MetadataBlock(notes=""),
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

    ctx_path = workdir / "outputs" / "round_0" / "round_ctx.json"
    assert ctx_path.exists()
    snap = json.loads(ctx_path.read_text())
    assert "core/data/x_dim" in snap
    assert "transform_x/identity/x_dim" in snap
    produced = snap.get("core/contracts/transform_x/identity/produced", [])
    assert "transform_x/identity/x_dim" in produced
    assert "model/random_forest/fit_metrics" in snap
    produced_model = snap.get("core/contracts/model/random_forest/produced", [])
    assert "model/random_forest/fit_metrics" in produced_model

    assert (workdir / "outputs" / "ledger.runs.parquet").exists()
    pred_dir = workdir / "outputs" / "ledger.predictions"
    assert pred_dir.exists()
    assert any(p.suffix == ".parquet" for p in pred_dir.iterdir())
