from __future__ import annotations

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.tfbs.stage_b.seed import (
    TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM,
    TFBS_STAGE_B_INITIAL_SEED_POLICY_UNIFORM_RANDOM,
    select_tfbs_stage_b_initial_ids,
)


def test_label_value_stratified_seed_samples_across_label_range() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(12)],
            "lexA_present": [0] * 6 + [1] * 6,
        }
    )

    selected = select_tfbs_stage_b_initial_ids(
        frame,
        label_name="lexA_present",
        initial_label_count=6,
        seed=7,
        policy=TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM,
    )

    selected_values = frame.set_index("id").loc[list(selected), "lexA_present"].tolist()
    assert selected_values.count(0) == 3
    assert selected_values.count(1) == 3


def test_uniform_random_seed_preserves_legacy_id_only_sampling() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"id-{idx}" for idx in range(12)],
            "lexA_present": [0] * 6 + [1] * 6,
        }
    )

    selected = select_tfbs_stage_b_initial_ids(
        frame,
        label_name="lexA_present",
        initial_label_count=6,
        seed=7,
        policy=TFBS_STAGE_B_INITIAL_SEED_POLICY_UNIFORM_RANDOM,
    )

    assert selected == ("id-3", "id-4", "id-6", "id-7", "id-8", "id-9")


def test_seed_selection_fails_fast_on_unknown_policy() -> None:
    frame = pd.DataFrame({"id": ["a", "b"], "lexA_present": [0, 1]})

    with pytest.raises(ValueError, match="Unsupported Stage B initial seed policy"):
        select_tfbs_stage_b_initial_ids(
            frame,
            label_name="lexA_present",
            initial_label_count=1,
            seed=7,
            policy="stress_batch0",
        )
