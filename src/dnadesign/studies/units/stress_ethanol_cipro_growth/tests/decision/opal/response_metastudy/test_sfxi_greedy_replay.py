"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_sfxi_greedy_replay.py

Contracts for the persisted historical SFXI greedy replay.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    SfxiEvidenceFrame,
    SfxiSourceProvenance,
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    build_historical_sfxi_greedy_replay,
)


def test_greedy_replay_binds_exact_persisted_top_k_and_overlap() -> None:
    evidence, scored = _fixture()

    replay = build_historical_sfxi_greedy_replay(evidence, scored, top_k=2)

    assert len(replay) == 6
    assert replay.groupby("selection_view_id")["id"].size().to_dict() == {
        "and": 2,
        "ciprofloxacin": 2,
        "ethanol": 2,
    }
    assert replay.groupby("selection_view_id")["rank"].apply(list).to_dict() == {
        "and": [1, 2],
        "ciprofloxacin": [1, 2],
        "ethanol": [1, 2],
    }
    assert replay["total_selection_slots"].eq(6).all()
    assert replay["unique_selected_sequences"].eq(3).all()
    assert replay["selected_in_all_views"].eq(1).all()
    assert replay["pairwise_overlap_total"].eq(4).all()
    assert replay.set_index(["selection_view_id", "rank"])["id"].to_dict() == {
        ("ethanol", 1): "shared-all",
        ("ethanol", 2): "ethanol-only",
        ("ciprofloxacin", 1): "shared-all",
        ("ciprofloxacin", 2): "cipro-and",
        ("and", 1): "shared-all",
        ("and", 2): "cipro-and",
    }
    assert replay.loc[replay["id"] == "shared-all", "selection_view_count"].eq(3).all()
    assert replay.loc[replay["id"] == "cipro-and", "selection_view_count"].eq(2).all()
    assert replay.loc[replay["id"] == "ethanol-only", "selection_view_count"].eq(1).all()
    assert replay.groupby("selection_view_id")["pool_candidate_count"].first().eq(5).all()
    assert replay["source_y_contract"].eq("sfxi_vec8").all()
    assert replay["evidence_lifecycle"].eq("provenance_only").all()
    assert replay["source_campaign_slug"].str.startswith("source-").all()
    assert replay["run_id"].str.startswith("run-").all()
    assert np.isfinite(replay["score_vs_effect_spearman"]).all()
    assert np.isfinite(replay["score_vs_logic_spearman"]).all()
    assert replay["effect_rank"].between(1, 5).all()
    assert replay["logic_rank"].between(1, 5).all()
    np.testing.assert_allclose(
        replay["score"].to_numpy(dtype=float),
        replay["logic_fidelity"].to_numpy(dtype=float) * replay["effect_scaled"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_greedy_replay_rejects_selection_or_score_drift() -> None:
    evidence, scored = _fixture()
    first = evidence[0]
    first.predictions.loc[first.predictions["id"] == "ethanol-only", "sel__is_selected"] = False
    first.predictions.loc[first.predictions["id"] == "cipro-and", "sel__is_selected"] = True
    first.predictions.loc[first.predictions["id"] == "cipro-and", "sel__rank_competition"] = 2

    with pytest.raises(ValueError, match="persisted selected identities"):
        build_historical_sfxi_greedy_replay(evidence, scored, top_k=2)

    evidence, scored = _fixture()
    scored["ethanol"].loc[scored["ethanol"]["id"] == "shared-all", "score"] += 0.01
    with pytest.raises(ValueError, match="logic fidelity multiplied by scaled effect"):
        build_historical_sfxi_greedy_replay(evidence, scored, top_k=2)


def _fixture() -> tuple[tuple[SfxiEvidenceFrame, ...], dict[str, pd.DataFrame]]:
    ids = ["shared-all", "ethanol-only", "cipro-and", "candidate-4", "candidate-5"]
    sequences = ["AAAA", "CCCC", "GGGG", "TTTT", "ACGT"]
    selected_by_view = {
        "ethanol": ("shared-all", "ethanol-only"),
        "ciprofloxacin": ("shared-all", "cipro-and"),
        "and": ("shared-all", "cipro-and"),
    }
    target_masks = {
        "ethanol": (0.0, 1.0, 0.0, 1.0),
        "ciprofloxacin": (0.0, 0.0, 1.0, 1.0),
        "and": (0.0, 0.0, 0.0, 1.0),
    }
    frames: list[SfxiEvidenceFrame] = []
    scored: dict[str, pd.DataFrame] = {}
    for view_index, view_id in enumerate(("ethanol", "ciprofloxacin", "and")):
        if view_id == "ethanol":
            logic = np.asarray([0.52, 0.45, 0.30, 0.28, 0.22], dtype=float)
            effect = np.asarray([0.80, 0.82, 0.75, 0.40, 0.30], dtype=float)
        else:
            logic = np.asarray([0.52, 0.30, 0.48, 0.28, 0.22], dtype=float) + view_index * 0.01
            effect = np.asarray([0.80, 0.70, 0.82, 0.40, 0.30], dtype=float) - view_index * 0.01
        score = logic * effect
        frame = pd.DataFrame(
            {
                "id": ids,
                "sequence": sequences,
                "selection_view_id": view_id,
                "score": score,
                "logic_fidelity": logic,
                "effect_scaled": effect,
            }
        ).sort_values(["score", "id"], ascending=[False, True], kind="mergesort")
        frame["rank"] = np.arange(1, len(frame) + 1)
        persisted_selected = selected_by_view[view_id]
        predictions = pd.DataFrame(
            {
                "id": ids,
                "sequence": sequences,
                "pred__y_hat_model": [np.zeros(8, dtype=float) for _ in ids],
                "pred__score_selected": score,
                "sel__rank_competition": [
                    persisted_selected.index(candidate) + 1 if candidate in persisted_selected else 3 + index
                    for index, candidate in enumerate(ids)
                ],
                "sel__is_selected": [candidate in persisted_selected for candidate in ids],
                "obj__logic_fidelity": logic,
                "obj__effect_scaled": effect,
            }
        )
        frames.append(
            SfxiEvidenceFrame(
                source=SfxiSourceProvenance(
                    source_id=f"source-{view_id}",
                    source_campaign_slug=f"source-{view_id}",
                    expected_run_id=f"run-{view_id}",
                    target_view_id=view_id,
                ),
                target_view=StressTargetView(
                    id=view_id,
                    label=view_id,
                    target_mask=target_masks[view_id],
                ),
                predictions=predictions,
                y_hat=np.zeros((len(ids), 8), dtype=float),
                denom=1.0,
                run_id=f"run-{view_id}",
            )
        )
        scored[view_id] = frame.reset_index(drop=True)
    return tuple(frames), scored
