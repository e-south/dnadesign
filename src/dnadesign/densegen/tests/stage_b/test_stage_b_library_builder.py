"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/stage_b/test_stage_b_library_builder.py

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

import dnadesign.densegen.src.core.pipeline.stage_b as stage_b_module
import dnadesign.densegen.src.core.pipeline.stage_b_library_builder as stage_b_library_builder_module
from dnadesign.densegen.src.config.generation import (
    FixedElements,
    PromoterConstraint,
    RegulatorConstraints,
    RegulatorGroup,
    ResolvedPlanItem,
    SamplingConfig,
)
from dnadesign.densegen.src.core.artifacts.library import LibraryRecord
from dnadesign.densegen.src.core.artifacts.pool import POOL_MODE_SEQUENCE, POOL_MODE_TFBS, PoolData
from dnadesign.densegen.src.core.pipeline.stage_b import _fixed_elements_label
from dnadesign.densegen.src.core.pipeline.stage_b_library_builder import LibraryBuilder


def test_build_library_for_plan_rejects_sequence_library_without_regulator_metadata() -> None:
    plan_item = ResolvedPlanItem(
        name="demo",
        quota=1,
        include_inputs=["demo"],
        fixed_elements=FixedElements(),
        regulator_constraints=RegulatorConstraints(groups=[]),
    )
    sampling_cfg = SamplingConfig(
        pool_strategy="subsample",
        library_source="build",
        library_size=2,
        library_sampling_strategy="tf_balanced",
    )
    pool = PoolData(
        name="demo",
        input_type="sequence_library",
        pool_mode=POOL_MODE_SEQUENCE,
        df=None,
        sequences=["AAAA", "CCCC", "GGGG"],
        pool_path=Path("."),
    )
    with pytest.raises(ValueError, match="requires cognate regulator metadata"):
        stage_b_module.build_library_for_plan(
            source_label="demo",
            plan_item=plan_item,
            pool=pool,
            sampling_cfg=sampling_cfg,
            seq_len=20,
            min_count_per_tf=0,
            usage_counts={},
            failure_counts=None,
            rng=random.Random(7),
            np_rng=np.random.default_rng(11),
            library_index_start=0,
        )


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


def test_library_builder_raises_when_library_selected_event_emit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        library_tfbs=["AAAA"],
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
        sequences=["AAAA"],
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
        events_path=tmp_path / "events.jsonl",
        library_build_rows=[],
        library_member_rows=[],
    )

    def _emit_event_fails(*_args, **_kwargs) -> None:
        raise RuntimeError("emit failed")

    monkeypatch.setattr(stage_b_library_builder_module, "_emit_event", _emit_event_fails)

    with pytest.raises(RuntimeError, match="Failed to emit LIBRARY_SELECTED event."):
        builder.build_next(library_index_start=0)


def test_record_library_build_raises_when_library_built_event_emit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_item = ResolvedPlanItem(
        name="demo",
        quota=1,
        include_inputs=["demo"],
        fixed_elements=FixedElements(),
        regulator_constraints=RegulatorConstraints(groups=[]),
    )
    sampling_cfg = SamplingConfig(
        pool_strategy="subsample",
        library_source="build",
        library_size=1,
        library_sampling_strategy="tf_balanced",
    )
    pool = PoolData(
        name="demo",
        input_type="binding_sites",
        pool_mode=POOL_MODE_TFBS,
        df=None,
        sequences=["AAAA"],
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
        library_source_label="build",
        library_records=None,
        library_cursor=None,
        events_path=tmp_path / "events.jsonl",
        library_build_rows=[],
        library_member_rows=[],
    )

    def _emit_event_fails(*_args, **_kwargs) -> None:
        raise RuntimeError("emit failed")

    monkeypatch.setattr(stage_b_library_builder_module, "_emit_event", _emit_event_fails)

    with pytest.raises(RuntimeError, match="Failed to emit LIBRARY_BUILT event."):
        builder._record_library_build(
            sampling_info={
                "library_index": 1,
                "library_hash": "hash",
                "library_size": 1,
                "pool_strategy": "subsample",
                "library_sampling_strategy": "tf_balanced",
            },
            library_tfbs=["AAAA"],
            library_tfs=["TF1"],
            library_tfbs_ids=[None],
            library_motif_ids=[None],
            library_site_ids=[None],
            library_sources=[None],
            fixed_bp=0,
            min_required_bp=0,
            slack_bp=0,
            infeasible=False,
            sequence_length=20,
        )


def test_record_library_build_raises_when_sampling_pressure_event_emit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_item = ResolvedPlanItem(
        name="demo",
        quota=1,
        include_inputs=["demo"],
        fixed_elements=FixedElements(),
        regulator_constraints=RegulatorConstraints(groups=[]),
    )
    sampling_cfg = SamplingConfig(
        pool_strategy="subsample",
        library_source="build",
        library_size=1,
        library_sampling_strategy="coverage_weighted",
    )
    pool = PoolData(
        name="demo",
        input_type="binding_sites",
        pool_mode=POOL_MODE_TFBS,
        df=None,
        sequences=["AAAA"],
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
        library_source_label="build",
        library_records=None,
        library_cursor=None,
        events_path=tmp_path / "events.jsonl",
        library_build_rows=[],
        library_member_rows=[],
    )

    def _emit_event_conditional(_path: Path, *, event: str, payload: dict) -> None:
        _ = payload
        if event == "LIBRARY_SAMPLING_PRESSURE":
            raise RuntimeError("emit failed")

    monkeypatch.setattr(stage_b_library_builder_module, "_emit_event", _emit_event_conditional)

    with pytest.raises(RuntimeError, match="Failed to emit LIBRARY_SAMPLING_PRESSURE event."):
        builder._record_library_build(
            sampling_info={
                "library_index": 1,
                "library_hash": "hash",
                "library_size": 1,
                "pool_strategy": "subsample",
                "library_sampling_strategy": "coverage_weighted",
                "sampling_weight_by_tf": {"TF1": 1.0},
                "sampling_weight_fraction_by_tf": {"TF1": 1.0},
                "sampling_usage_count_by_tf": {"TF1": 0},
                "sampling_failure_count_by_tf": {"TF1": 0},
            },
            library_tfbs=["AAAA"],
            library_tfs=["TF1"],
            library_tfbs_ids=[None],
            library_motif_ids=[None],
            library_site_ids=[None],
            library_sources=[None],
            fixed_bp=0,
            min_required_bp=0,
            slack_bp=0,
            infeasible=False,
            sequence_length=20,
        )


def _coverage_weighted_test_context() -> tuple[ResolvedPlanItem, SamplingConfig, PoolData]:
    plan_item = ResolvedPlanItem(
        name="demo_plan",
        quota=2,
        include_inputs=["demo_input"],
        fixed_elements=FixedElements(),
        regulator_constraints=RegulatorConstraints(groups=[]),
    )
    sampling_cfg = SamplingConfig(
        pool_strategy="subsample",
        library_source="build",
        library_size=1,
        library_sampling_strategy="coverage_weighted",
        avoid_failed_motifs=True,
        unique_binding_sites=True,
        unique_binding_cores=True,
    )
    pool_df = pd.DataFrame(
        {
            "tf": ["TF1", "TF1"],
            "tfbs": ["AAAA", "CCCC"],
            "tfbs_core": ["AAAA", "CCCC"],
        }
    )
    pool = PoolData(
        name="demo_input",
        input_type="binding_sites",
        pool_mode=POOL_MODE_TFBS,
        df=pool_df,
        sequences=pool_df["tfbs"].tolist(),
        pool_path=Path("."),
    )
    return plan_item, sampling_cfg, pool


def test_build_library_for_plan_uses_failure_feedback_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSampler:
        calls: list[dict[tuple[str, str], int]] = []

        def __init__(self, df: pd.DataFrame, rng: np.random.Generator) -> None:
            self._df = df
            self._rng = rng

        def generate_binding_site_library(self, library_size: int, **kwargs):
            _ = (library_size, self._df, self._rng)
            failure_counts = dict(kwargs.get("failure_counts") or {})
            _FakeSampler.calls.append(failure_counts)
            site = "CCCC" if failure_counts.get(("TF1", "AAAA"), 0) > 0 else "AAAA"
            return [site], [f"TF1:{site}"], ["TF1"], {"achieved_length": len(site)}

    monkeypatch.setattr(stage_b_module, "TFSampler", _FakeSampler)

    plan_item, sampling_cfg, pool = _coverage_weighted_test_context()

    library1, _parts1, _labels1, _info1 = stage_b_module.build_library_for_plan(
        source_label="demo_input",
        plan_item=plan_item,
        pool=pool,
        sampling_cfg=sampling_cfg,
        seq_len=10,
        min_count_per_tf=0,
        usage_counts={},
        failure_counts={},
        rng=random.Random(1),
        np_rng=np.random.default_rng(2),
        library_index_start=0,
    )
    assert library1 == ["AAAA"]

    failure_counts = {
        ("demo_input", "demo_plan", "TF1", "AAAA", None): {"required_regulators": 3},
        ("other_input", "demo_plan", "TF1", "AAAA", None): {"required_regulators": 99},
    }
    library2, _parts2, _labels2, _info2 = stage_b_module.build_library_for_plan(
        source_label="demo_input",
        plan_item=plan_item,
        pool=pool,
        sampling_cfg=sampling_cfg,
        seq_len=10,
        min_count_per_tf=0,
        usage_counts={},
        failure_counts=failure_counts,
        rng=random.Random(1),
        np_rng=np.random.default_rng(2),
        library_index_start=1,
    )
    assert library2 == ["CCCC"]

    assert _FakeSampler.calls[0] == {}
    assert _FakeSampler.calls[1] == {("TF1", "AAAA"): 3}


def test_build_library_for_plan_passes_usage_counts_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSampler:
        calls: list[dict[tuple[str, str], int]] = []

        def __init__(self, df: pd.DataFrame, rng: np.random.Generator) -> None:
            self._df = df
            self._rng = rng

        def generate_binding_site_library(self, library_size: int, **kwargs):
            _ = (library_size, self._df, self._rng)
            usage_counts = dict(kwargs.get("usage_counts") or {})
            _FakeSampler.calls.append(usage_counts)
            site = "CCCC" if usage_counts.get(("TF1", "AAAA"), 0) > 0 else "AAAA"
            return [site], [f"TF1:{site}"], ["TF1"], {"achieved_length": len(site)}

    monkeypatch.setattr(stage_b_module, "TFSampler", _FakeSampler)
    plan_item, sampling_cfg, pool = _coverage_weighted_test_context()

    usage_counts: dict[tuple[str, str], int] = {}
    library1, _parts1, _labels1, _info1 = stage_b_module.build_library_for_plan(
        source_label="demo_input",
        plan_item=plan_item,
        pool=pool,
        sampling_cfg=sampling_cfg,
        seq_len=10,
        min_count_per_tf=0,
        usage_counts=usage_counts,
        failure_counts={},
        rng=random.Random(1),
        np_rng=np.random.default_rng(2),
        library_index_start=0,
    )
    assert library1 == ["AAAA"]

    usage_counts[("TF1", "AAAA")] = 5
    library2, _parts2, _labels2, _info2 = stage_b_module.build_library_for_plan(
        source_label="demo_input",
        plan_item=plan_item,
        pool=pool,
        sampling_cfg=sampling_cfg,
        seq_len=10,
        min_count_per_tf=0,
        usage_counts=usage_counts,
        failure_counts={},
        rng=random.Random(1),
        np_rng=np.random.default_rng(2),
        library_index_start=1,
    )
    assert library2 == ["CCCC"]

    assert _FakeSampler.calls[0] == {}
    assert _FakeSampler.calls[1] == {("TF1", "AAAA"): 5}
