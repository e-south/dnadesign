"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_observed_sfxi_runtime.py

Runtime boundaries for historical observed-label SFXI evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.opal import SFXIScoringConfig, score_vec8
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.core.contracts import (
    SfxiEvidenceFrame,
    SfxiSourceProvenance,
    StressTargetView,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    candidate_identity,
    observed_sfxi,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime.label_truth import (
    LabelTruthState,
)

_VEC8_COLUMNS = ("v00", "v10", "v01", "v11", "y00_star", "y10_star", "y01_star", "y11_star")


def test_historical_replay_uses_persisted_six_slot_contract_without_label_truth(tmp_path: Path) -> None:
    source = _source_rows()
    labels = pd.DataFrame(
        {
            "id": source["id"],
            "sequence": source["sequence"],
            "observed_round": 0,
            "y_obs": [row.copy() for row in source.loc[:, _VEC8_COLUMNS].to_numpy(dtype=float)],
        }
    )
    contexts = _evidence_frames(source)
    absent = tmp_path / "not-needed-when-label-truth-is-absent.parquet"
    bindings = candidate_identity.ResponseCandidateIdentityBindings(
        rows=pd.DataFrame(),
        bundle_root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        records_path=absent,
        binding_count=0,
        candidate_count=0,
        excluded_design_count=0,
    )

    result = observed_sfxi.build_historical_observed_sfxi_evidence(
        source,
        labels,
        sfxi_evidence=contexts,
        label_truth_state=LabelTruthState(
            state="not_ready",
            label_source_state="not_verified",
            observed_label_promotion_manifest=None,
        ),
        candidate_bindings=bindings,
    )

    assert not absent.exists()
    assert result.components.groupby("selection_view_id")["is_highest_observed_sfxi"].sum().eq(6).all()
    assert result.components["in_promoted_response_window_corpus"].isna().all()


def _source_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(8):
        rows.append(
            {
                "id": f"candidate-{index}",
                "sequence": f"ACGT-{index}",
                "design_id": "pDual-10-spyp" if index == 6 else "pDual-10-sulAp" if index == 7 else f"ES{index}",
                "reader_experiment_id": f"experiment-{index % 3}",
                "v00": 0.10 + index * 0.02,
                "v10": 0.70 - index * 0.01,
                "v01": 0.20 + index * 0.01,
                "v11": 0.85 - index * 0.01,
                "y00_star": -1.5 + index * 0.11,
                "y10_star": -0.8 + index * 0.15,
                "y01_star": -1.2 + index * 0.09,
                "y11_star": -0.4 + index * 0.18,
            }
        )
    return pd.DataFrame.from_records(rows)


def _evidence_frames(source: pd.DataFrame) -> tuple[SfxiEvidenceFrame, ...]:
    vec8 = source.loc[:, _VEC8_COLUMNS].to_numpy(dtype=float)
    specs = (
        ("ethanol", (0.0, 1.0, 0.0, 1.0)),
        ("ciprofloxacin", (0.0, 0.0, 1.0, 1.0)),
        ("and", (0.0, 0.0, 0.0, 1.0)),
    )
    return tuple(
        SfxiEvidenceFrame(
            source=SfxiSourceProvenance(
                source_id=f"source-{view_id}",
                source_campaign_slug=f"campaign-{view_id}",
                expected_run_id=f"run-{view_id}",
                target_view_id=view_id,
            ),
            target_view=StressTargetView(id=view_id, label=view_id, target_mask=mask),
            predictions=pd.DataFrame(),
            y_hat=np.empty((0, 8)),
            denom=float(score_vec8(vec8, SFXIScoringConfig(setpoint_vector=mask)).denom_used),
            run_id=f"run-{view_id}",
        )
        for view_id, mask in specs
    )
