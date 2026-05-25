"""
Shared label-source runtime planning contracts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.opal.src.config.types import (
    CampaignBlock,
    CandidateScope,
    DataBlock,
    IngestBlock,
    LabelsBlock,
    LabelSourceUSRSidecar,
    LocationUSR,
    ObjectivesBlock,
    PluginRef,
    RootConfig,
    SafetyBlock,
    ScoringBlock,
    SelectionBlock,
    TrainingBlock,
    WritebackBlock,
)
from dnadesign.opal.src.runtime.round_plan import plan_round
from dnadesign.opal.src.runtime.run_round import RunRoundRequest, run_round
from dnadesign.opal.src.storage.data_access import RecordsStore
from dnadesign.opal.src.storage.label_sources import label_source_from_config, label_source_status
from dnadesign.opal.src.storage.state import CampaignState


def _records(tmp_path: Path) -> tuple[pd.DataFrame, Path, Path]:
    usr_root = tmp_path / "usr" / "datasets"
    dataset_root = usr_root / "demo_candidates"
    dataset_root.mkdir(parents=True)
    records_path = dataset_root / "records.parquet"
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "sequence": ["AAA", "BBB", "CCC"],
            "bio_type": ["dna", "dna", "dna"],
            "alphabet": ["dna_4", "dna_4", "dna_4"],
            "X": [[0.1], [0.2], [0.3]],
        }
    )
    df.to_parquet(records_path, index=False)
    return df, usr_root, records_path


def _labels(usr_root: Path) -> None:
    path = usr_root / "demo_candidates" / "_opal" / "observed_labels.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "observed_round": [0, 1],
            "batch_id": ["batch0", "batch1"],
            "y_space": ["sfxi_vec8", "sfxi_vec8"],
            "y_obs": [[0.0], [1.0]],
            "src": ["assay", "assay"],
            "ts": ["2026-05-17T00:00:00Z", "2026-05-17T00:01:00Z"],
        }
    ).to_parquet(path, index=False)


def _cfg(usr_root: Path, slug: str, *, top_k: int) -> RootConfig:
    return RootConfig(
        campaign=CampaignBlock(name=slug, slug=slug, workdir=str(usr_root / ".." / slug)),
        data=DataBlock(
            location=LocationUSR(kind="usr", path=str(usr_root), dataset="demo_candidates"),
            x_column_name="X",
            y_column_name=f"opal__{slug}__y",
            transforms_x=PluginRef(name="identity", params={}),
            transforms_y=PluginRef(name="scalar_from_table_v1", params={}),
            y_expected_length=1,
        ),
        labels=LabelsBlock(
            source=LabelSourceUSRSidecar(
                kind="usr_sidecar",
                dataset="demo_candidates",
                path="_opal/observed_labels.parquet",
            ),
            y_space="sfxi_vec8",
            id_column="id",
            round_column="observed_round",
            batch_column="batch_id",
            dedup_policy="latest_by_round",
        ),
        model=PluginRef(name="random_forest", params={"n_estimators": 5, "random_state": 0}),
        selection=SelectionBlock(
            selection=PluginRef(
                name="top_n",
                params={
                    "top_k": top_k,
                    "score_ref": "scalar_identity_v1/scalar",
                    "objective_mode": "maximize",
                    "tie_handling": "competition_rank",
                },
            )
        ),
        objectives=ObjectivesBlock(objectives=[PluginRef(name="scalar_identity_v1", params={})]),
        training=TrainingBlock(policy={"cumulative_training": True}),
        ingest=IngestBlock(duplicate_policy="error"),
        scoring=ScoringBlock(score_batch_size=1000),
        safety=SafetyBlock(),
        writeback=WritebackBlock(prediction_records="ledger_only"),
    )


def _store(cfg: RootConfig, records_path: Path) -> RecordsStore:
    return RecordsStore(
        kind="usr",
        records_path=records_path,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params=cfg.data.transforms_x.params,
    )


def _write_state(cfg: RootConfig, records_path: Path) -> None:
    workdir = Path(cfg.campaign.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    CampaignState(
        campaign_slug=cfg.campaign.slug,
        campaign_name=cfg.campaign.name,
        workdir=str(workdir.resolve()),
        data_location={
            "kind": "usr",
            "path": str(Path(cfg.data.location.path).resolve()),
            "dataset": cfg.data.location.dataset,
            "records_path": str(records_path.resolve()),
        },
        x_column_name=cfg.data.x_column_name,
        y_column_name=cfg.data.y_column_name,
    ).save(workdir / "state.json")


def test_plan_round_uses_shared_labels_and_selector_specific_top_k(tmp_path: Path) -> None:
    df, usr_root, records_path = _records(tmp_path)
    _labels(usr_root)
    store = RecordsStore(
        kind="usr",
        records_path=records_path,
        campaign_slug="eth",
        x_col="X",
        y_col="opal__eth__y",
        x_transform_name="identity",
        x_transform_params={},
    )

    eth_cfg = _cfg(usr_root, "eth", top_k=1)
    cipro_cfg = _cfg(usr_root, "cipro", top_k=2)
    eth_source = label_source_from_config(eth_cfg, store)
    cipro_source = label_source_from_config(cipro_cfg, store)

    eth_plan = plan_round(store, df, eth_cfg, 1, label_source=eth_source)
    cipro_plan = plan_round(store, df, cipro_cfg, 1, label_source=cipro_source)

    assert eth_plan.training_df["id"].tolist() == ["a", "b"]
    assert cipro_plan.training_df["id"].tolist() == ["a", "b"]
    assert eth_cfg.selection.selection.params["top_k"] == 1
    assert cipro_cfg.selection.selection.params["top_k"] == 2
    assert eth_plan.candidate_df["id"].tolist() == ["c"]
    assert cipro_plan.candidate_df["id"].tolist() == ["c"]


def test_candidate_scope_limits_candidate_universe_without_copying_records(tmp_path: Path) -> None:
    df, usr_root, records_path = _records(tmp_path)
    _labels(usr_root)
    scope_path = tmp_path / "scope.parquet"
    pd.DataFrame({"id": ["c", "b"]}).to_parquet(scope_path, index=False)
    cfg = _cfg(usr_root, "scoped", top_k=1)
    cfg.data.candidate_scope = CandidateScope(kind="id_list", path=str(scope_path), id_column="id")
    store = _store(cfg, records_path)
    source = label_source_from_config(cfg, store)

    plan = plan_round(store, df, cfg, 1, label_source=source)

    assert plan.candidate_total_before_filter == 2
    assert plan.candidate_df["id"].astype(str).tolist() == ["c"]


def test_plan_round_rejects_train_eval_overlap_when_exclusion_is_configured(tmp_path: Path) -> None:
    df, usr_root, records_path = _records(tmp_path)
    store = _store(_cfg(usr_root, "eth", top_k=1), records_path)
    cfg = _cfg(usr_root, "eth", top_k=1)

    class _BadLabelSource:
        kind = "test_bad_source"

        def training_labels(self, df, as_of_round, *, cumulative_training, dedup_policy):
            _unused = (df, as_of_round, cumulative_training, dedup_policy)
            return pd.DataFrame({"id": ["a"], "y": [[1.0]], "r": [0]})

        def labeled_id_set_leq_round(self, df, as_of_round):
            _unused = (df, as_of_round)
            return set()

        def labeled_id_set_any_round(self, df):
            _unused = df
            return set()

    try:
        plan_round(store, df, cfg, 0, label_source=_BadLabelSource())
    except Exception as exc:
        message = str(exc)
    else:
        raise AssertionError("expected train/eval overlap to fail fast")

    assert "LeakageContractError" in message
    assert "train_eval_overlap" in message


def test_run_round_shared_labels_keep_selection_ledgers_campaign_local(tmp_path: Path) -> None:
    df, usr_root, records_path = _records(tmp_path)
    _labels(usr_root)
    eth_cfg = _cfg(usr_root, "eth", top_k=1)
    cipro_cfg = _cfg(usr_root, "cipro", top_k=2)
    _write_state(eth_cfg, records_path)
    _write_state(cipro_cfg, records_path)

    eth_res = run_round(
        _store(eth_cfg, records_path),
        df.copy(),
        RunRoundRequest(cfg=eth_cfg, as_of_round=1, verbose=False),
    )
    cipro_res = run_round(
        _store(cipro_cfg, records_path),
        pd.read_parquet(records_path),
        RunRoundRequest(cfg=cipro_cfg, as_of_round=1, verbose=False),
    )

    assert eth_res.trained_on == 2
    assert cipro_res.trained_on == 2
    assert eth_res.top_k_requested == 1
    assert cipro_res.top_k_requested == 2
    assert eth_res.ledger_path != cipro_res.ledger_path
    assert Path(eth_res.ledger_path).exists()
    assert Path(cipro_res.ledger_path).exists()
    records_after = pd.read_parquet(records_path)
    assert "opal__eth__label_hist" not in records_after.columns
    assert "opal__cipro__label_hist" not in records_after.columns


def test_run_round_shared_sidecar_rejects_current_y_contamination(tmp_path: Path) -> None:
    df, usr_root, records_path = _records(tmp_path)
    _labels(usr_root)
    cfg = _cfg(usr_root, "eth", top_k=1)
    _write_state(cfg, records_path)
    contaminated = df.copy()
    contaminated[cfg.data.y_column_name] = [[1.0], None, None]

    try:
        run_round(
            _store(cfg, records_path),
            contaminated,
            RunRoundRequest(cfg=cfg, as_of_round=1, verbose=False),
        )
    except Exception as exc:
        message = str(exc)
    else:
        raise AssertionError("expected shared-sidecar current-Y contamination to fail fast")

    assert "LeakageContractError" in message
    assert "records_y_column_contamination" in message

    status = label_source_status(cfg, _store(cfg, records_path), contaminated, strict=False)
    assert status["valid"] is False
    assert status["leakage"]["status"] == "fail"
    assert status["leakage"]["violations"][0]["code"] == "records_y_column_contamination"


def test_run_round_shared_sidecar_rejects_ledger_only_label_history_contamination(tmp_path: Path) -> None:
    df, usr_root, records_path = _records(tmp_path)
    _labels(usr_root)
    cfg = _cfg(usr_root, "eth", top_k=1)
    _write_state(cfg, records_path)
    contaminated = df.copy()
    contaminated[f"opal__{cfg.campaign.slug}__label_hist"] = [
        [
            {
                "kind": "label",
                "observed_round": 0,
                "y_obs": {"value": [1.0], "dtype": "vector", "schema": {"length": 1}},
                "src": "stale_campaign_history",
            }
        ],
        [],
        [],
    ]

    try:
        run_round(
            _store(cfg, records_path),
            contaminated,
            RunRoundRequest(cfg=cfg, as_of_round=1, verbose=False),
        )
    except Exception as exc:
        message = str(exc)
    else:
        raise AssertionError("expected shared-sidecar label-history contamination to fail fast")

    assert "LeakageContractError" in message
    assert "records_label_history_contamination" in message
