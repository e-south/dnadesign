# ABOUTME: Validates run_round correctness guards around objective outputs.
# ABOUTME: Exercises run_round behavior that should fail fast on invalid scores.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_run_round_integrity.py

Module Author(s): Eric J. South (extended by Codex)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

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
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.registries.objectives import list_objectives, register_objective
from dnadesign.opal.src.runtime.run_round import RunRoundRequest, run_round
from dnadesign.opal.src.storage.data_access import RecordsStore
from dnadesign.opal.src.storage.state import CampaignState
from dnadesign.opal.src.transforms_x import identity  # noqa: F401 (registers identity)


class _ObjectiveResult:
    def __init__(self, score: np.ndarray, diagnostics: Optional[Dict[str, Any]] = None, mode: str = "maximize") -> None:
        self.score = score
        self.diagnostics = diagnostics or {}
        self.mode = mode


def _register_bad_length_objective() -> str:
    name = "test_bad_length_v1"
    if name in list_objectives():
        return name

    @register_objective(name)
    def _bad_length_objective(*, y_pred, params: Dict[str, Any], ctx=None, train_view=None) -> _ObjectiveResult:
        _ = y_pred, params, ctx, train_view
        return _ObjectiveResult(score=np.asarray([0.1]), diagnostics={})

    return name


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
    selection_params: Optional[Dict[str, Any]] = None,
) -> RootConfig:
    return RootConfig(
        campaign=CampaignBlock(name="Demo", slug="demo", workdir=str(workdir)),
        data=DataBlock(
            location=LocationLocal(kind="local", path=str(records_path)),
            x_column_name="X",
            y_column_name="Y",
            transforms_x=PluginRef(name="identity", params={}),
            transforms_y=PluginRef(name="scalar_from_table_v1", params={}),
            y_expected_length=1,
        ),
        model=PluginRef(name="random_forest", params={"n_estimators": 5, "random_state": 0}),
        selection=SelectionBlock(selection=PluginRef(name="top_n", params=selection_params or {"top_k": 1})),
        objective=ObjectiveBlock(objective=PluginRef(name=objective_name, params=objective_params or {})),
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

    with pytest.raises(OpalError, match="Objective produced .* scores for .* candidates"):
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

    sel_path = workdir / "outputs" / "rounds" / "round_0" / "selection" / "selection_top_k.parquet"
    assert sel_path.exists()
    sel_df = pd.read_parquet(sel_path)
    assert pd.isna(sel_df.loc[0, "sequence"])
