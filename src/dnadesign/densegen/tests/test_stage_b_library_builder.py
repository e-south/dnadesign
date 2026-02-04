"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_stage_b_library_builder.py

Stage-B library builder validation tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from dnadesign.densegen.src.config.generation import (
    FixedElements,
    RegulatorConstraints,
    RegulatorGroup,
    ResolvedPlanItem,
    SamplingConfig,
)
from dnadesign.densegen.src.core.artifacts.library import LibraryRecord
from dnadesign.densegen.src.core.artifacts.pool import POOL_MODE_TFBS, PoolData
from dnadesign.densegen.src.core.pipeline.stage_b_library_builder import LibraryBuilder


def test_library_builder_requires_required_regulators_for_groups() -> None:
    plan_item = ResolvedPlanItem(
        name="demo",
        quota=10,
        include_inputs=["demo"],
        fixed_elements=FixedElements(),
        regulator_constraints=RegulatorConstraints(
            groups=[RegulatorGroup(name="group", members=["TF1"], min_required=1)]
        ),
    )
    sampling_cfg = SamplingConfig(
        pool_strategy="full",
        library_source="artifact",
        library_artifact_path="artifact",
        library_size=2,
        library_sampling_strategy="tf_balanced",
    )
    record = LibraryRecord(
        input_name="demo",
        plan_name="demo",
        library_index=1,
        library_hash="hash",
        library_id="hash",
        library_tfbs=["AAA", "CCC"],
        library_tfs=["TF1", "TF1"],
        library_site_ids=[None, None],
        library_sources=[None, None],
        library_tfbs_ids=[None, None],
        library_motif_ids=[None, None],
        pool_strategy="full",
        library_sampling_strategy="tf_balanced",
        library_size=2,
        achieved_length=None,
        relaxed_cap=None,
        final_cap=None,
        iterative_max_libraries=None,
        iterative_min_new_solutions=None,
        required_regulators_selected=None,
    )
    pool = PoolData(
        name="demo",
        input_type="binding_sites",
        pool_mode=POOL_MODE_TFBS,
        df=None,
        sequences=["AAA", "CCC"],
        pool_path=Path("."),
    )
    builder = LibraryBuilder(
        source_label="demo",
        plan_item=plan_item,
        pool=pool,
        sampling_cfg=sampling_cfg,
        seq_len=10,
        min_count_per_tf=0,
        usage_counts={},
        failure_counts={},
        rng=random.Random(1),
        np_rng=np.random.default_rng(2),
        library_source_label="artifact",
        library_records={("demo", "demo"): [record]},
        library_cursor={},
        events_path=None,
        library_build_rows=[],
        library_member_rows=[],
    )

    with pytest.raises(RuntimeError, match="required_regulators_selected"):
        builder.build_next(library_index_start=0)
