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
import pandas as pd
import pytest

from dnadesign.densegen.src.config.generation import (
    FixedElements,
    PromoterConstraint,
    RegulatorConstraints,
    RegulatorGroup,
    ResolvedPlanItem,
    SamplingConfig,
)
from dnadesign.densegen.src.core.artifacts.library import LibraryRecord
from dnadesign.densegen.src.core.artifacts.pool import POOL_MODE_TFBS, PoolData
from dnadesign.densegen.src.core.pipeline.stage_b import _fixed_elements_label
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


def test_fixed_elements_label_joins_names() -> None:
    fixed = FixedElements(
        promoter_constraints=[
            PromoterConstraint(name="sigma70_consensus"),
            PromoterConstraint(name="sigma54"),
            PromoterConstraint(name="sigma70_consensus"),
        ]
    )
    assert _fixed_elements_label(fixed) == "sigma70_consensus+sigma54"


def test_library_builder_allows_short_library_bp() -> None:
    plan_item = ResolvedPlanItem(
        name="demo",
        quota=1,
        include_inputs=["demo"],
        fixed_elements=FixedElements(),
        regulator_constraints=RegulatorConstraints(
            groups=[RegulatorGroup(name="group", members=["TF1", "TF2"], min_required=1)]
        ),
    )
    sampling_cfg = SamplingConfig(
        pool_strategy="full",
        library_source="build",
        library_size=2,
        library_sampling_strategy="tf_balanced",
    )
    df = pd.DataFrame(
        {
            "tf": ["TF1", "TF2"],
            "tfbs": ["A" * 10, "T" * 10],
            "tfbs_core": ["A" * 10, "T" * 10],
        }
    )
    pool = PoolData(
        name="demo",
        input_type="binding_sites",
        pool_mode=POOL_MODE_TFBS,
        df=df,
        sequences=df["tfbs"].tolist(),
        pool_path=Path("."),
    )
    builder = LibraryBuilder(
        source_label="demo",
        plan_item=plan_item,
        pool=pool,
        sampling_cfg=sampling_cfg,
        seq_len=60,
        min_count_per_tf=0,
        usage_counts={},
        failure_counts=None,
        rng=random.Random(1),
        np_rng=np.random.default_rng(2),
        library_source_label="build",
        library_records=None,
        library_cursor=None,
        events_path=None,
        library_build_rows=[],
        library_member_rows=[],
    )

    context = builder.build_next(library_index_start=0)

    assert context.sampling_info["achieved_length"] == 20


def test_library_builder_allows_infeasible_min_required_len() -> None:
    plan_item = ResolvedPlanItem(
        name="demo",
        quota=1,
        include_inputs=["demo"],
        fixed_elements=FixedElements(
            promoter_constraints=[
                PromoterConstraint(
                    name="sigma70",
                    upstream="TTGACA",
                    downstream="TATAAT",
                    spacer_length=(16, 20),
                )
            ]
        ),
        regulator_constraints=RegulatorConstraints(
            groups=[RegulatorGroup(name="response", members=["TF1", "TF2"], min_required=2)]
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
        library_tfbs=["A" * 18, "T" * 16],
        library_tfs=["TF1", "TF2"],
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
        required_regulators_selected=["TF1", "TF2"],
    )
    pool = PoolData(
        name="demo",
        input_type="binding_sites",
        pool_mode=POOL_MODE_TFBS,
        df=None,
        sequences=["A" * 18, "T" * 16],
        pool_path=Path("."),
    )
    builder = LibraryBuilder(
        source_label="demo",
        plan_item=plan_item,
        pool=pool,
        sampling_cfg=sampling_cfg,
        seq_len=60,
        min_count_per_tf=0,
        usage_counts={},
        failure_counts=None,
        rng=random.Random(1),
        np_rng=np.random.default_rng(2),
        library_source_label="artifact",
        library_records={("demo", "demo"): [record]},
        library_cursor={},
        events_path=None,
        library_build_rows=[],
        library_member_rows=[],
    )

    context = builder.build_next(library_index_start=0)

    assert context.infeasible is True
    assert context.min_required_len > 60


def test_library_builder_allows_long_motif_length() -> None:
    plan_item = ResolvedPlanItem(
        name="demo",
        quota=1,
        include_inputs=["demo"],
        fixed_elements=FixedElements(),
        regulator_constraints=RegulatorConstraints(groups=[]),
    )
    sampling_cfg = SamplingConfig(
        pool_strategy="full",
        library_source="artifact",
        library_artifact_path="artifact",
        library_size=1,
        library_sampling_strategy="tf_balanced",
    )
    record = LibraryRecord(
        input_name="demo",
        plan_name="demo",
        library_index=1,
        library_hash="hash",
        library_id="hash",
        library_tfbs=["A" * 25],
        library_tfs=["TF1"],
        library_site_ids=[None],
        library_sources=[None],
        library_tfbs_ids=[None],
        library_motif_ids=[None],
        pool_strategy="full",
        library_sampling_strategy="tf_balanced",
        library_size=1,
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
        sequences=["A" * 25],
        pool_path=Path("."),
    )
    builder = LibraryBuilder(
        source_label="demo",
        plan_item=plan_item,
        pool=pool,
        sampling_cfg=sampling_cfg,
        seq_len=20,
        min_count_per_tf=0,
        usage_counts={},
        failure_counts=None,
        rng=random.Random(1),
        np_rng=np.random.default_rng(2),
        library_source_label="artifact",
        library_records={("demo", "demo"): [record]},
        library_cursor={},
        events_path=None,
        library_build_rows=[],
        library_member_rows=[],
    )

    context = builder.build_next(library_index_start=0)

    assert context.min_required_len == 0
    assert context.infeasible is False
