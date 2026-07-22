"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/yiu/test_payload_rendering.py

Runtime contracts for the payload-centric YIU v4 lane.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

import dnadesign.cruncher.app.yiu_workflow.render as yiu_workflow_render_module
import dnadesign.cruncher.yiu.optimizer as yiu_optimizer_module
import dnadesign.cruncher.yiu.render as yiu_render_module
from dnadesign.cruncher.app.yiu_workflow.render import render_yiu_spec, render_yiu_spec_outcome
from dnadesign.cruncher.app.yiu_workflow.show import show_yiu_bundle
from dnadesign.cruncher.bio import reverse_complement_iupac
from dnadesign.cruncher.yiu.bsmbi import build_split_fragment_display_specs
from dnadesign.cruncher.yiu.bundle_models import PayloadViewEntry, PayloadVisualInventory
from dnadesign.cruncher.yiu.bundle_paths import resolve_composite_render_artifact_path
from dnadesign.cruncher.yiu.bundle_surface import YiuShowOutcome
from dnadesign.cruncher.yiu.candidate_generation import CandidatePlan, MutationChoice, enumerate_candidates
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload, build_ligation_search_state
from dnadesign.cruncher.yiu.errors import NoFeasiblePlanError, YiuContractError
from dnadesign.cruncher.yiu.load import load_yiu_spec
from dnadesign.cruncher.yiu.mismatch_notation import compact_mismatch_notation_groups
from dnadesign.cruncher.yiu.normalize import normalize_payload
from dnadesign.cruncher.yiu.optimizer import select_best_candidate
from dnadesign.cruncher.yiu.publish_inventory import (
    build_normalized_payload_dump,
    build_payload_bundle_manifest,
    build_payload_visual_inventory,
)
from dnadesign.cruncher.yiu.publish_io import (
    write_debug_render_jobs,
    write_normalized_payload_dump,
    write_payload_bundle_state,
    write_payload_bundle_views,
)
from dnadesign.cruncher.yiu.publish_layout import resolve_payload_bundle_layout
from dnadesign.cruncher.yiu.render_panels import (
    YIU_NUCLEOTIDE_LEGEND_CANONICAL_COLOR,
    YIU_NUCLEOTIDE_LEGEND_MISMATCH_COLOR,
    _draw_composite_nucleotide_legend,
    save_composite_render,
)
from dnadesign.cruncher.yiu.scoring import CandidateScore
from dnadesign.cruncher.yiu.spec_models import MismatchesSpec
from dnadesign.cruncher.yiu.view_catalog import build_payload_view_entries, build_render_job_payload
from dnadesign.cruncher.yiu.view_contracts import (
    build_assembled_payload_view_contract,
    build_split_payload_view_rows,
    build_yiu_style_overrides,
)
from dnadesign.cruncher.yiu.view_payload_content import (
    build_payload_mismatch_annotations,
    build_payload_motif_layers,
    build_payload_view_meta,
)
from dnadesign.cruncher.yiu.view_payload_contracts import build_payload_view_contract
from dnadesign.cruncher.yiu.view_registry import validate_payload_view_entry
from dnadesign.cruncher.yiu.view_sequence_metadata import (
    build_assembled_payload_view_meta,
    build_split_payload_row_meta,
)
from dnadesign.cruncher.yiu.view_styles import get_yiu_style_profile

TOY_SEQUENCE = "AAATTTCCCGGGAAATTTCCC"
TOY_JUNCTION_START = 4
TOY_JUNCTION_END = 8
SECONDARY_OBJECTIVES = [
    "total_loss",
    "ligation_awareness",
    "midpoint_proximity",
    "default_strand_preference",
    "lexical_stability",
]


def test_validate_payload_view_entry_rejects_registry_drift() -> None:
    entry = PayloadViewEntry(
        view_id="payload",
        visual_direction="evidence_ribbon",
        contract_kind="sequence_evidence_map_v1",
        input_kind="json",
        view_contract_path="payload_view.json",
        render_artifact_path="payload_views.pdf",
        renderer_kind="nucleotide_evidence_map",
    )

    with pytest.raises(ValueError, match="YIU view entry drift"):
        validate_payload_view_entry(entry)


def test_validate_payload_view_entry_rejects_visual_direction_drift() -> None:
    entry = PayloadViewEntry(
        view_id="payload",
        visual_direction="operator_strip",
        contract_kind="yiu_payload_visual_v1",
        input_kind="json",
        view_contract_path="payload_view.json",
        render_artifact_path="payload_views.pdf",
        renderer_kind="nucleotide_evidence_map",
    )

    with pytest.raises(ValueError, match="visual_direction"):
        validate_payload_view_entry(entry)


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_sample_csv(path: Path, *, hit_id: str, sequence: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["elite_id", "elite_rank", "elite_sequence"])
        writer.writeheader()
        writer.writerow({"elite_id": hit_id, "elite_rank": 1, "elite_sequence": sequence})


def _write_sample_parquet(
    path: Path,
    *,
    hit_id: str,
    sequence: str,
    per_tf_json: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, object] = {"id": hit_id, "sequence": sequence, "rank": 1}
    if per_tf_json is not None:
        row["per_tf_json"] = json.dumps(per_tf_json)
    pd.DataFrame([row]).to_parquet(path, index=False)


def _write_occurrences_parquet(path: Path, *, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_sample_pwm_config(path: Path, *, tf_rows: dict[str, list[list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cruncher": {
            "pwms_info": {
                tf_name: {
                    "consensus": "N" * len(rows),
                    "pwm_matrix": rows,
                }
                for tf_name, rows in tf_rows.items()
            }
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _pwm_rows(preferred_base: str, *, width: int = 4) -> list[list[float]]:
    rows: list[list[float]] = []
    index = {"A": 0, "C": 1, "G": 2, "T": 3}[preferred_base]
    for _ in range(width):
        row = [0.01, 0.01, 0.01, 0.01]
        row[index] = 0.97
        rows.append(row)
    return rows


def _rounded_pwm_rows(preferred_base: str, *, width: int = 4) -> list[list[float]]:
    rows = _pwm_rows(preferred_base, width=width)
    rounded: list[list[float]] = []
    for row in rows:
        rounded.append([round(float(item), 2) for item in row])
    return rounded


def _canonical_tetr_pwm_rows() -> list[list[float]]:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "tetr_pwm_rows.json"
    return json.loads(path.read_text(encoding="utf-8"))["matrix"]


def _inline_pwm_context() -> dict[str, object]:
    return {
        "contract": "yiu_pwm_context_v1",
        "schema_version": 1,
        "name": "inline_pwm_context",
        "motifs": [
            {
                "motif_instance_id": "motif_plus",
                "tf_name": "TF_PLUS",
                "motif_name": "plus",
                "reference_strand": "+",
                "start": 3,
                "end": 7,
                "probabilities": {"alphabet": ["A", "C", "G", "T"], "rows": _pwm_rows("T")},
                "provenance": {"source_kind": "inline", "source_ref": "test-inline"},
            },
            {
                "motif_instance_id": "motif_minus",
                "tf_name": "TF_MINUS",
                "motif_name": "minus",
                "reference_strand": "-",
                "start": 5,
                "end": 9,
                "probabilities": {"alphabet": ["A", "C", "G", "T"], "rows": _pwm_rows("A")},
                "provenance": {"source_kind": "inline", "source_ref": "test-inline"},
            },
        ],
    }


def _tetr_pwm_context() -> dict[str, object]:
    return {
        "contract": "yiu_pwm_context_v1",
        "schema_version": 1,
        "name": "tetr_monotypic_pwm_context",
        "motifs": [
            {
                "motif_instance_id": "tetR_payload_site",
                "tf_name": "tetR",
                "motif_name": "tetr_demo",
                "reference_strand": "+",
                "start": 0,
                "end": 17,
                "probabilities": {
                    "alphabet": ["A", "C", "G", "T"],
                    "rows": _canonical_tetr_pwm_rows(),
                },
                "provenance": {
                    "source_kind": "file",
                    "source_ref": "tests/fixtures/tetr_pwm_rows.json",
                },
            }
        ],
    }


def _junction_payload(
    *,
    mode: str = "explicit_window",
    start: int = TOY_JUNCTION_START,
    end: int = TOY_JUNCTION_END,
    max_payload_body_length: int | None = None,
) -> dict[str, object]:
    effective_max_payload_body_length = (
        13 if max_payload_body_length is None and mode == "explicit_window" else max_payload_body_length or 12
    )
    payload: dict[str, object] = {
        "mode": mode,
        "overhang_length": 4,
        "max_payload_body_length": effective_max_payload_body_length,
    }
    if mode == "explicit_window":
        payload["start"] = start
        payload["end"] = end
    return payload


def _pwm_payload(
    *,
    mode: str = "none",
    source: dict[str, object] | None = None,
) -> dict[str, object]:
    effective_source = {"kind": "none"} if mode == "none" else (source or {"kind": "none"})
    return {
        "mode": mode,
        "source": effective_source,
        "objective": {
            "primary": "maximin",
            "secondary": list(SECONDARY_OBJECTIVES),
        },
    }


def _user_sequence_spec(
    *,
    name: str = "demo_payload",
    sequence: str = TOY_SEQUENCE,
    junction_mode: str = "explicit_window",
    junction_start: int = TOY_JUNCTION_START,
    junction_end: int = TOY_JUNCTION_END,
    mismatch_count: int = 1,
    candidate_positions: list[int] | None = None,
    allowed_strands: list[str] | None = None,
    ligation_profile: str = "none",
    ligation_awareness_mode: str = "secondary",
    bad_pattern_heuristics: bool = False,
    ligation_selection_mode: str | None = None,
    pwm_worst_loss_tolerance: float | None = None,
    pwm_total_loss_tolerance: float | None = None,
    max_worst_mismatch_class_tier: int | None = None,
    max_middle_mismatch_count: int | None = None,
    allow_double_middle: bool | None = None,
    allow_tnna_like_overhangs: bool | None = None,
    pwm_mode: str = "none",
    pwm_source: dict[str, object] | None = None,
    emit_render_jobs_debug: bool = False,
) -> dict[str, object]:
    effective_allowed_strands = ["complement", "payload"] if allowed_strands is None else allowed_strands
    mismatches: dict[str, object] = {
        "count": mismatch_count,
        "allowed_strands": effective_allowed_strands,
        "strand_mode": "per_position",
        "default_strand_preference": "complement",
        "ligation_profile": ligation_profile,
        "ligation_awareness_mode": ligation_awareness_mode,
        "bad_pattern_heuristics": bad_pattern_heuristics,
    }
    if ligation_selection_mode is not None:
        mismatches["ligation_selection_mode"] = ligation_selection_mode
    if pwm_worst_loss_tolerance is not None:
        mismatches["pwm_worst_loss_tolerance"] = pwm_worst_loss_tolerance
    if pwm_total_loss_tolerance is not None:
        mismatches["pwm_total_loss_tolerance"] = pwm_total_loss_tolerance
    if max_worst_mismatch_class_tier is not None:
        mismatches["max_worst_mismatch_class_tier"] = max_worst_mismatch_class_tier
    if max_middle_mismatch_count is not None:
        mismatches["max_middle_mismatch_count"] = max_middle_mismatch_count
    if allow_double_middle is not None:
        mismatches["allow_double_middle"] = allow_double_middle
    if allow_tnna_like_overhangs is not None:
        mismatches["allow_tnna_like_overhangs"] = allow_tnna_like_overhangs
    if candidate_positions is not None:
        mismatches["candidate_positions"] = candidate_positions
    return {
        "yiu": {
            "schema_version": 1,
            "contract": "split_yiu_payload_rendering_v4",
            "name": name,
        },
        "input": {
            "kind": "user_sequence",
            "user_sequence": {"sequence": sequence},
        },
        "optimization": {
            "junction": _junction_payload(mode=junction_mode, start=junction_start, end=junction_end),
            "mismatches": mismatches,
            "pwm": _pwm_payload(mode=pwm_mode, source=pwm_source),
        },
        "output": {
            "bundle_dir": f"outputs/{name}",
            "published_plot_path": f"outputs/plot__{name}__payload_views.pdf",
            "emit_render_jobs_debug": emit_render_jobs_debug,
        },
    }


def _sample_hit_spec(
    *,
    name: str = "sample_hit_payload",
    hit_id: str = "elite-1",
    sample_name: str = "sample",
    payload_sequence: str | None = None,
    source_artifact_path: str | None = None,
    metadata: dict[str, object] | None = None,
    junction_mode: str = "center_locked",
    junction_start: int = TOY_JUNCTION_START,
    junction_end: int = TOY_JUNCTION_END,
    mismatch_count: int = 1,
    candidate_positions: list[int] | None = None,
    allowed_strands: list[str] | None = None,
    ligation_profile: str = "none",
    ligation_awareness_mode: str = "secondary",
    bad_pattern_heuristics: bool = False,
    ligation_selection_mode: str | None = None,
    pwm_worst_loss_tolerance: float | None = None,
    pwm_total_loss_tolerance: float | None = None,
    max_worst_mismatch_class_tier: int | None = None,
    max_middle_mismatch_count: int | None = None,
    allow_double_middle: bool | None = None,
    allow_tnna_like_overhangs: bool | None = None,
    pwm_mode: str = "none",
    pwm_source: dict[str, object] | None = None,
) -> dict[str, object]:
    sample_hit: dict[str, object] = {"hit_id": hit_id, "sample_name": sample_name}
    if payload_sequence is not None:
        sample_hit["payload_sequence"] = payload_sequence
    if source_artifact_path is not None:
        sample_hit["source_artifact_path"] = source_artifact_path
    if metadata is not None:
        sample_hit["metadata"] = metadata
    effective_allowed_strands = ["complement", "payload"] if allowed_strands is None else allowed_strands
    mismatches: dict[str, object] = {
        "count": mismatch_count,
        "allowed_strands": effective_allowed_strands,
        "strand_mode": "per_position",
        "default_strand_preference": "complement",
        "ligation_profile": ligation_profile,
        "ligation_awareness_mode": ligation_awareness_mode,
        "bad_pattern_heuristics": bad_pattern_heuristics,
    }
    if ligation_selection_mode is not None:
        mismatches["ligation_selection_mode"] = ligation_selection_mode
    if pwm_worst_loss_tolerance is not None:
        mismatches["pwm_worst_loss_tolerance"] = pwm_worst_loss_tolerance
    if pwm_total_loss_tolerance is not None:
        mismatches["pwm_total_loss_tolerance"] = pwm_total_loss_tolerance
    if max_worst_mismatch_class_tier is not None:
        mismatches["max_worst_mismatch_class_tier"] = max_worst_mismatch_class_tier
    if max_middle_mismatch_count is not None:
        mismatches["max_middle_mismatch_count"] = max_middle_mismatch_count
    if allow_double_middle is not None:
        mismatches["allow_double_middle"] = allow_double_middle
    if allow_tnna_like_overhangs is not None:
        mismatches["allow_tnna_like_overhangs"] = allow_tnna_like_overhangs
    if candidate_positions is not None:
        mismatches["candidate_positions"] = candidate_positions
    return {
        "yiu": {
            "schema_version": 1,
            "contract": "split_yiu_payload_rendering_v4",
            "name": name,
        },
        "input": {
            "kind": "sample_hit",
            "sample_hit": sample_hit,
        },
        "optimization": {
            "junction": _junction_payload(mode=junction_mode, start=junction_start, end=junction_end),
            "mismatches": mismatches,
            "pwm": _pwm_payload(mode=pwm_mode, source=pwm_source),
        },
        "output": {
            "bundle_dir": f"outputs/yiu__{name}",
            "published_plot_path": f"outputs/plots/plot__yiu__{name}__payload_views.pdf",
            "emit_render_jobs_debug": False,
        },
    }


def _legacy_v1_payload_spec() -> dict[str, object]:
    return {
        "yiu": {
            "schema_version": 1,
            "contract": "split_yiu_payload_rendering_v1",
            "name": "legacy_payload",
        },
        "input": {
            "kind": "user_sequence",
            "user_sequence": {"sequence": TOY_SEQUENCE},
        },
        "split": {"mode": "derived"},
        "bulge_mask": {"positions": [1]},
        "output": {
            "bundle_dir": "outputs/legacy_payload",
            "emit_render_jobs_debug": False,
        },
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate(
    *,
    lexical_key: str,
    default_strand_preference_count: int,
    midpoint_distance: int = 0,
    mismatch_positions: tuple[int, ...] = (1,),
    mutations: tuple[MutationChoice, ...] | None = None,
) -> CandidatePlan:
    resolved_mutations = mutations or (
        MutationChoice(
            junction_offset=mismatch_positions[0],
            payload_index=4 + mismatch_positions[0],
            mutated_strand="complement",
            native_base="A",
            mutated_base="C",
            opposing_base="T",
        ),
    )
    return CandidatePlan(
        junction_start=4,
        junction_end=8,
        mismatch_positions=mismatch_positions,
        mutations=resolved_mutations,
        midpoint_distance=midpoint_distance,
        middle_mismatch_count=sum(1 for item in mismatch_positions if item in {1, 2}),
        double_middle_flag=len(mismatch_positions) == 2 and all(item in {1, 2} for item in mismatch_positions),
        default_strand_preference_count=default_strand_preference_count,
        lexical_key=lexical_key,
    )


def _select_result(
    *,
    candidates: tuple[CandidatePlan, ...],
    pwm_effective: bool,
    ligation_profile: str = "none",
    ligation_awareness_mode: str = "disabled",
    bad_pattern_heuristics: bool = False,
    ligation_selection_mode: str = "secondary",
    pwm_worst_loss_tolerance: float = 0.0,
    pwm_total_loss_tolerance: float = 0.0,
    max_worst_mismatch_class_tier: int = 0,
    max_middle_mismatch_count: int = 1,
    allow_double_middle: bool = False,
    allow_tnna_like_overhangs: bool = False,
) -> object:
    ligation_state = build_ligation_search_state(
        ligation_profile=ligation_profile,
        ligation_awareness_mode=ligation_awareness_mode,
        ligation_selection_mode=ligation_selection_mode,
        candidate_positions=sorted({position for candidate in candidates for position in candidate.mismatch_positions}),
        pwm_worst_loss_tolerance=pwm_worst_loss_tolerance,
        pwm_total_loss_tolerance=pwm_total_loss_tolerance,
        max_worst_mismatch_class_tier=max_worst_mismatch_class_tier,
        max_middle_mismatch_count=max_middle_mismatch_count,
        allow_double_middle=allow_double_middle,
        allow_tnna_like_overhangs=allow_tnna_like_overhangs,
    )
    return select_best_candidate(
        candidates=candidates,
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
        scorable_motifs=(),
        pwm_effective=pwm_effective,
        ligation_state=ligation_state,
        bad_pattern_heuristics=bad_pattern_heuristics,
    )


def test_load_yiu_spec_rejects_legacy_contract(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "legacy.yiu.yaml"
    _write_yaml(spec_path, _legacy_v1_payload_spec())

    with pytest.raises(YiuContractError, match="YIU_CONTRACT_UNKNOWN"):
        load_yiu_spec(spec_path)


def test_load_yiu_spec_rejects_invalid_overhang_length(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "invalid_overhang.yiu.yaml"
    payload = _user_sequence_spec()
    payload["optimization"]["junction"]["overhang_length"] = 5
    _write_yaml(spec_path, payload)

    with pytest.raises(ValueError, match="optimization.junction.overhang_length"):
        load_yiu_spec(spec_path)


@pytest.mark.parametrize(
    "bundle_dir",
    ["/tmp/escape", "../escape"],
)
def test_load_yiu_spec_rejects_workspace_escaping_bundle_dir(tmp_path: Path, bundle_dir: str) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "escape.yiu.yaml"
    payload = _user_sequence_spec()
    payload["output"]["bundle_dir"] = bundle_dir
    _write_yaml(spec_path, payload)

    with pytest.raises(ValueError, match="output.bundle_dir"):
        load_yiu_spec(spec_path)


def test_normalize_user_sequence_explicit_window_keeps_reference_payload_when_complement_mutates(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "toy_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec(candidate_positions=[1]))

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    assert normalized.contract == "yiu_normalized_payload_v5"
    assert normalized.reference_complement_sequence == "".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE)
    assert normalized.selected_payload_sequence == TOY_SEQUENCE
    assert normalized.junction.start == TOY_JUNCTION_START
    assert normalized.junction.end == TOY_JUNCTION_END
    assert normalized.junction.mode == "explicit_window"
    assert len(normalized.mismatches) == 1
    mismatch = normalized.mismatches[0]
    assert mismatch.payload_index == TOY_JUNCTION_START + 1
    assert mismatch.junction_offset == 1
    assert mismatch.mutated_strand == "complement"
    assert normalized.selected_complement_sequence[mismatch.payload_index] == mismatch.mutated_base


def test_normalize_payload_persists_ligation_rationale_fields(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "ligation_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="ligation_payload",
            candidate_positions=[0],
            ligation_profile="t4",
            ligation_awareness_mode="secondary",
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    assert normalized.ligation_profile == "t4"
    assert normalized.ligation_awareness_mode == "secondary"
    assert normalized.chosen_ligation_key is not None
    assert normalized.ligation_rationale
    assert normalized.ligation_rationale[0].position_class == "edge"
    assert normalized.ligation_rationale[0].canonical_mismatch_class == "GT"


def test_load_yiu_spec_rejects_removed_derived_junction_mode(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "derived_payload.yiu.yaml"
    _write_yaml(
        spec_path, _user_sequence_spec(sequence="AACCGGTTGGTT", junction_mode="derived", candidate_positions=[1])
    )

    with pytest.raises(ValueError, match="optimization\\.junction\\.mode"):
        load_yiu_spec(spec_path)


def test_load_yiu_spec_rejects_removed_source_artifact_alias(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "source_alias_payload.yiu.yaml"
    payload = _sample_hit_spec(payload_sequence="AACCGGTTGGTT", candidate_positions=[1])
    payload["input"]["sample_hit"]["source_artifact"] = "outputs/optimize/tables/elites.parquet"
    _write_yaml(spec_path, payload)

    with pytest.raises(ValueError, match="source_artifact"):
        load_yiu_spec(spec_path)


def test_normalize_center_locked_mode_selects_midpoint_nearest_internal_window(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "center_locked_payload.yiu.yaml"
    _write_yaml(
        spec_path, _user_sequence_spec(sequence="AACCGGTTGGTT", junction_mode="center_locked", candidate_positions=[1])
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    assert normalized.junction.mode == "center_locked"
    assert normalized.junction.start == 4
    assert normalized.junction.end == 8


def test_normalize_center_locked_mode_raises_when_no_window_satisfies_body_bound(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "center_locked_body_bound.yiu.yaml"
    payload = _user_sequence_spec(sequence="AACCGGTTGGTT", junction_mode="center_locked")
    payload["optimization"]["junction"]["max_payload_body_length"] = 3
    _write_yaml(spec_path, payload)

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    with pytest.raises(NoFeasiblePlanError, match="No feasible junction window found"):
        normalize_payload(spec, workspace_root=workspace_root)


def test_normalize_explicit_window_raises_when_body_bound_is_exceeded(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "explicit_body_bound.yiu.yaml"
    payload = _user_sequence_spec(
        sequence="AACCGGTTAA",
        junction_mode="explicit_window",
        junction_start=1,
        junction_end=5,
        candidate_positions=[1],
    )
    payload["optimization"]["junction"]["max_payload_body_length"] = 4
    _write_yaml(spec_path, payload)

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    with pytest.raises(YiuContractError, match="max_payload_body_length=4"):
        normalize_payload(spec, workspace_root=workspace_root)


def test_normalize_optimize_mode_raises_no_feasible_plan_for_tight_body_bound(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "no_plan.yiu.yaml"
    payload = _user_sequence_spec(sequence="AACCGGTTA", junction_mode="optimize")
    payload["optimization"]["junction"]["max_payload_body_length"] = 1
    _write_yaml(spec_path, payload)

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    with pytest.raises(NoFeasiblePlanError, match="No feasible optimized junction found"):
        normalize_payload(spec, workspace_root=workspace_root)


def test_normalize_payload_raises_when_payload_is_too_short_for_any_internal_window(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "short_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(sequence="AACCG", junction_mode="center_locked", candidate_positions=[1]),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    with pytest.raises(YiuContractError, match="too short for any internal 4-nt junction window"):
        normalize_payload(spec, workspace_root=workspace_root)


def test_enumerate_candidates_without_pwm_enumerates_all_non_native_bases() -> None:
    mismatches_spec = MismatchesSpec.model_validate(
        {
            "count": 1,
            "candidate_positions": [1],
            "allowed_strands": ["complement", "payload"],
            "strand_mode": "per_position",
            "default_strand_preference": "complement",
        }
    )

    candidates = enumerate_candidates(
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
        junction_starts=(TOY_JUNCTION_START,),
        mismatches_spec=mismatches_spec,
    )

    assert len(candidates) == 6
    mutated_bases_by_strand = {
        strand: {
            candidate.mutations[0].mutated_base
            for candidate in candidates
            if candidate.mutations[0].mutated_strand == strand
        }
        for strand in ("complement", "payload")
    }
    assert mutated_bases_by_strand == {
        "complement": {"C", "G", "T"},
        "payload": {"A", "C", "G"},
    }


def test_candidate_generation_count_unchanged() -> None:
    legacy_spec = MismatchesSpec.model_validate(
        {
            "count": 2,
            "candidate_positions": [0, 1, 2, 3],
            "allowed_strands": ["complement", "payload"],
            "strand_mode": "per_position",
            "default_strand_preference": "complement",
        }
    )
    ligation_spec = MismatchesSpec.model_validate(
        {
            "count": 2,
            "candidate_positions": [0, 1, 2, 3],
            "allowed_strands": ["complement", "payload"],
            "strand_mode": "per_position",
            "default_strand_preference": "complement",
            "ligation_profile": "t4",
            "ligation_awareness_mode": "secondary",
            "bad_pattern_heuristics": False,
        }
    )

    legacy_candidates = enumerate_candidates(
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
        junction_starts=(TOY_JUNCTION_START,),
        mismatches_spec=legacy_spec,
    )
    ligation_candidates = enumerate_candidates(
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
        junction_starts=(TOY_JUNCTION_START,),
        mismatches_spec=ligation_spec,
    )

    assert len(ligation_candidates) == len(legacy_candidates)


def test_bad_pattern_heuristics_disabled_by_default() -> None:
    spec = MismatchesSpec.model_validate(
        {
            "count": 1,
            "candidate_positions": [0, 1, 2, 3],
            "allowed_strands": ["complement", "payload"],
            "strand_mode": "per_position",
            "default_strand_preference": "complement",
        }
    )

    assert spec.bad_pattern_heuristics is False


def test_mismatches_spec_defaults_omitted_candidate_positions_to_all_junction_offsets() -> None:
    spec = MismatchesSpec.model_validate({"count": 1})

    assert spec.candidate_positions == [0, 1, 2, 3]


def test_mismatches_spec_rejects_empty_candidate_position_pool() -> None:
    with pytest.raises(ValueError, match="candidate_positions must be non-empty"):
        MismatchesSpec.model_validate({"count": 1, "candidate_positions": []})


def test_mismatches_spec_rejects_count_larger_than_candidate_position_pool() -> None:
    with pytest.raises(ValueError, match="exceeds the candidate position pool size"):
        MismatchesSpec.model_validate({"count": 2, "candidate_positions": [0]})


def test_enumerate_candidates_supports_all_junction_offsets() -> None:
    mismatches_spec = MismatchesSpec.model_validate(
        {
            "count": 1,
            "candidate_positions": [0, 1, 2, 3],
            "allowed_strands": ["complement", "payload"],
            "strand_mode": "per_position",
            "default_strand_preference": "complement",
        }
    )

    candidates = enumerate_candidates(
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
        junction_starts=(TOY_JUNCTION_START,),
        mismatches_spec=mismatches_spec,
    )

    assert {candidate.mutations[0].junction_offset for candidate in candidates} == {0, 1, 2, 3}


def test_normalize_sample_hit_resolves_csv_and_enforces_payload_assertion(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    export_table = workspace / "sample_outputs" / "export" / "table__elites.csv"
    _write_sample_csv(export_table, hit_id="elite-1", sequence="TTGGAACCAA")

    good_spec_path = workspace / "configs" / "yiu" / "sample_hit_payload.yiu.yaml"
    _write_yaml(
        good_spec_path,
        _sample_hit_spec(
            payload_sequence="TTGGAACCAA",
            source_artifact_path=str(export_table.relative_to(workspace)),
            candidate_positions=[1],
        ),
    )
    spec, _resolved_spec_path, workspace_root = load_yiu_spec(good_spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    assert normalized.input_kind == "sample_hit"
    assert normalized.reference_payload_sequence == "TTGGAACCAA"
    assert normalized.source_provenance["source_artifact_path"] == str(export_table.resolve())

    bad_spec_path = workspace / "configs" / "yiu" / "sample_hit_mismatch.yiu.yaml"
    _write_yaml(
        bad_spec_path,
        _sample_hit_spec(
            name="sample_hit_mismatch",
            payload_sequence="AACCGGTTAA",
            source_artifact_path=str(export_table.relative_to(workspace)),
            candidate_positions=[1],
        ),
    )
    bad_spec, _resolved_bad_path, workspace_root = load_yiu_spec(bad_spec_path)
    with pytest.raises(YiuContractError, match="payload_sequence does not match"):
        normalize_payload(bad_spec, workspace_root=workspace_root)


def test_load_yiu_spec_rejects_ambiguous_user_sequence_iupac_payload(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "ambiguous_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec(sequence="AANNNNTT"))

    with pytest.raises(ValueError, match="YIU_SEQUENCE_INVALID"):
        load_yiu_spec(spec_path)


def test_normalize_sample_hit_rejects_ambiguous_source_sequence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    export_table = workspace / "sample_outputs" / "export" / "table__elites.csv"
    _write_sample_csv(export_table, hit_id="elite-1", sequence="AANNNNTT")

    spec_path = workspace / "configs" / "yiu" / "sample_hit_ambiguous_source.yiu.yaml"
    _write_yaml(
        spec_path,
        _sample_hit_spec(
            source_artifact_path=str(export_table.relative_to(workspace)),
            candidate_positions=[1],
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    with pytest.raises(YiuContractError, match="exact A/C/G/T bases for YIU v4"):
        normalize_payload(spec, workspace_root=workspace_root)


def test_normalize_sample_hit_resolves_sibling_workspace_from_parent_root(tmp_path: Path) -> None:
    workspaces_root = tmp_path / "workspaces"
    sibling_workspace = workspaces_root / "demo_monotypic_tetr"
    artifact_path = sibling_workspace / "outputs" / "optimize" / "tables" / "elites.parquet"
    _write_sample_parquet(
        artifact_path,
        hit_id="demo_monotypic_tetr_elite_001",
        sequence="CTCTATATCTGATATAGAG",
    )

    yiu_workspace = workspaces_root / "yiu_workspace"
    spec_path = yiu_workspace / "configs" / "yiu" / "sample_hit_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _sample_hit_spec(
            hit_id="demo_monotypic_tetr_elite_001",
            sample_name="tetr_monotypic",
            source_artifact_path="outputs/optimize/tables/elites.parquet",
            metadata={
                "source_workspace": "demo_monotypic_tetr",
            },
            candidate_positions=[1],
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    assert normalized.reference_payload_sequence == "CTCTATATCTGATATAGAG"
    assert normalized.source_provenance["source_workspace"] == str(sibling_workspace.resolve())


def test_normalize_sample_hit_missing_sibling_workspace_fails_fast(tmp_path: Path) -> None:
    yiu_workspace = tmp_path / "workspaces" / "yiu_workspace"
    spec_path = yiu_workspace / "configs" / "yiu" / "sample_hit_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _sample_hit_spec(
            hit_id="missing_elite_001",
            sample_name="missing_workspace",
            source_artifact_path="outputs/optimize/tables/elites.parquet",
            metadata={
                "source_workspace": "missing_workspace",
            },
            candidate_positions=[1],
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    with pytest.raises(YiuContractError, match="sample-hit source workspace not found"):
        normalize_payload(spec, workspace_root=workspace_root)


def test_normalize_pwm_use_if_available_records_fallback_reason(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "missing_pwm.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="missing_pwm",
            pwm_mode="use_if_available",
            pwm_source={"kind": "file", "path": "motifs/missing_pwm_context.yaml"},
            candidate_positions=[1],
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    assert normalized.motif_context.effective is False
    assert normalized.motif_context.source_kind == "file"
    assert normalized.motif_context.fallback_reason is not None
    assert "missing_pwm_context.yaml" in normalized.motif_context.fallback_reason


def test_normalize_pwm_require_fails_when_sample_context_is_unavailable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    artifact_path = workspace / "sample_outputs" / "export" / "table__elites.parquet"
    _write_sample_parquet(artifact_path, hit_id="elite-1", sequence="TTGGAACCAA")

    spec_path = workspace / "configs" / "yiu" / "required_pwm.yiu.yaml"
    _write_yaml(
        spec_path,
        _sample_hit_spec(
            source_artifact_path=str(artifact_path.relative_to(workspace)),
            pwm_mode="require",
            pwm_source={"kind": "sample_context"},
            candidate_positions=[1],
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    with pytest.raises(YiuContractError, match="YIU_PWM_CONTEXT_REQUIRED"):
        normalize_payload(spec, workspace_root=workspace_root)


def test_normalize_sample_context_builds_effective_multi_motif_context(tmp_path: Path) -> None:
    sibling_workspace = tmp_path / "demo_monotypic_tetr"
    artifact_path = sibling_workspace / "outputs" / "optimize" / "tables" / "elites.parquet"
    _write_sample_parquet(
        artifact_path,
        hit_id="demo_monotypic_tetr_elite_001",
        sequence="CTCTATATCTGATATAGAG",
        per_tf_json={
            "TF_PLUS": {"best_start": 6, "width": 4, "strand": "+", "motif_name": "plus"},
            "TF_MINUS": {"best_start": 8, "width": 4, "strand": "-", "motif_name": "minus"},
        },
    )
    _write_sample_pwm_config(
        sibling_workspace / "outputs" / "meta" / "config_used.yaml",
        tf_rows={
            "TF_PLUS": _pwm_rows("T"),
            "TF_MINUS": _pwm_rows("A"),
        },
    )

    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "sample_context.yiu.yaml"
    _write_yaml(
        spec_path,
        _sample_hit_spec(
            hit_id="demo_monotypic_tetr_elite_001",
            sample_name="tetr_monotypic",
            source_artifact_path=str(artifact_path.resolve()),
            pwm_mode="require",
            pwm_source={"kind": "sample_context"},
            mismatch_count=2,
            candidate_positions=[1, 2],
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    assert normalized.motif_context.effective is True
    assert {motif.reference_strand for motif in normalized.motif_context.motifs} == {"+", "-"}
    assert {motif.tf_name for motif in normalized.motif_context.motifs} == {"TF_PLUS", "TF_MINUS"}


def test_normalize_sample_context_renormalizes_rounded_pwm_rows(tmp_path: Path) -> None:
    sibling_workspace = tmp_path / "demo_monotypic_tetr"
    artifact_path = sibling_workspace / "outputs" / "optimize" / "tables" / "elites.parquet"
    _write_sample_parquet(
        artifact_path,
        hit_id="demo_monotypic_tetr_elite_rounded",
        sequence="CTCTATATCTGATATAGAG",
        per_tf_json={
            "TF_ROUNDED": {"best_start": 0, "width": 4, "strand": "+", "motif_name": "rounded"},
        },
    )
    _write_sample_pwm_config(
        sibling_workspace / "outputs" / "meta" / "config_used.yaml",
        tf_rows={"TF_ROUNDED": _rounded_pwm_rows("T")},
    )

    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "sample_context_rounded.yiu.yaml"
    _write_yaml(
        spec_path,
        _sample_hit_spec(
            hit_id="demo_monotypic_tetr_elite_rounded",
            sample_name="tetr_monotypic",
            source_artifact_path=str(artifact_path.resolve()),
            pwm_mode="require",
            pwm_source={"kind": "sample_context"},
            mismatch_count=1,
            candidate_positions=[1],
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    assert normalized.motif_context.effective is True
    motif = normalized.motif_context.motifs[0]
    for row in motif.probabilities.rows:
        assert sum(row) == pytest.approx(1.0)
        assert row[3] > 0.96


def test_normalize_sample_context_prefers_selected_occurrence_rows_for_multiplicity_payloads(tmp_path: Path) -> None:
    sibling_workspace = tmp_path / "demo_monotypic_baer"
    artifact_path = sibling_workspace / "outputs" / "optimize" / "tables" / "elites.parquet"
    occurrences_path = sibling_workspace / "outputs" / "optimize" / "tables" / "elites_occurrences.parquet"
    _write_sample_parquet(
        artifact_path,
        hit_id="demo_monotypic_baer_elite_001",
        sequence="TTTTTTCGCGAAAAAA",
        per_tf_json={
            "baeR": {
                "best_start": 0,
                "width": 11,
                "strand": "+",
                "motif_name": "representative_only",
            }
        },
    )
    _write_occurrences_parquet(
        occurrences_path,
        rows=[
            {
                "elite_id": "demo_monotypic_baer_elite_001",
                "tf": "baeR",
                "occurrence_rank": 1,
                "start": 0,
                "end": 11,
                "strand": "+",
                "selected": True,
            },
            {
                "elite_id": "demo_monotypic_baer_elite_001",
                "tf": "baeR",
                "occurrence_rank": 2,
                "start": 2,
                "end": 13,
                "strand": "+",
                "selected": True,
            },
            {
                "elite_id": "demo_monotypic_baer_elite_001",
                "tf": "baeR",
                "occurrence_rank": 3,
                "start": 3,
                "end": 14,
                "strand": "-",
                "selected": True,
            },
        ],
    )
    _write_sample_pwm_config(
        sibling_workspace / "outputs" / "meta" / "config_used.yaml",
        tf_rows={"baeR": _pwm_rows("T", width=11)},
    )

    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "sample_context_occurrences.yiu.yaml"
    _write_yaml(
        spec_path,
        _sample_hit_spec(
            hit_id="demo_monotypic_baer_elite_001",
            sample_name="baer_monotypic",
            source_artifact_path=str(artifact_path.resolve()),
            pwm_mode="require",
            pwm_source={"kind": "sample_context"},
            mismatch_count=2,
            candidate_positions=[1, 2],
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    assert normalized.motif_context.effective is True
    assert len(normalized.motif_context.motifs) == 3
    assert [(motif.start, motif.end, motif.reference_strand) for motif in normalized.motif_context.motifs] == [
        (0, 11, "+"),
        (2, 13, "+"),
        (3, 14, "-"),
    ]


def test_select_best_candidate_breaks_pwm_tie_on_total_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_a = _candidate(lexical_key="a", default_strand_preference_count=1)
    candidate_b = _candidate(lexical_key="b", default_strand_preference_count=1)

    def _stub_score_candidate(*, candidate, reference_payload_sequence, reference_complement_sequence, scorable_motifs):
        _ = (reference_payload_sequence, reference_complement_sequence, scorable_motifs)
        if candidate.lexical_key == "a":
            return CandidateScore(worst_loss=1.0, total_loss=2.0)
        return CandidateScore(worst_loss=1.0, total_loss=1.0)

    monkeypatch.setattr(yiu_optimizer_module, "score_candidate", _stub_score_candidate)
    result = _select_result(candidates=(candidate_a, candidate_b), pwm_effective=True)

    assert result.winner.lexical_key == "b"
    assert result.score.total_loss == 1.0


def test_select_best_candidate_breaks_pwm_tie_on_midpoint_proximity(monkeypatch: pytest.MonkeyPatch) -> None:
    farther = _candidate(
        lexical_key="farther",
        default_strand_preference_count=1,
        midpoint_distance=2,
    )
    closer = _candidate(
        lexical_key="closer",
        default_strand_preference_count=1,
        midpoint_distance=1,
    )

    monkeypatch.setattr(
        yiu_optimizer_module,
        "score_candidate",
        lambda **_: CandidateScore(worst_loss=1.0, total_loss=1.0),
    )
    result = _select_result(candidates=(farther, closer), pwm_effective=True)

    assert result.winner.lexical_key == "closer"


def test_select_best_candidate_breaks_full_tie_on_lexical_key(monkeypatch: pytest.MonkeyPatch) -> None:
    lexical_first = _candidate(
        lexical_key="a",
        default_strand_preference_count=1,
        midpoint_distance=1,
    )
    lexical_second = _candidate(
        lexical_key="b",
        default_strand_preference_count=1,
        midpoint_distance=1,
    )

    monkeypatch.setattr(
        yiu_optimizer_module,
        "score_candidate",
        lambda **_: CandidateScore(worst_loss=1.0, total_loss=1.0),
    )
    result = _select_result(candidates=(lexical_second, lexical_first), pwm_effective=True)

    assert result.winner.lexical_key == "a"


def test_ligation_profile_none_preserves_legacy_order() -> None:
    edge_candidate = _candidate(
        lexical_key="edge",
        default_strand_preference_count=1,
        mismatch_positions=(0,),
        mutations=(
            MutationChoice(
                junction_offset=0,
                payload_index=4,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
        ),
    )
    middle_candidate = _candidate(
        lexical_key="middle",
        default_strand_preference_count=1,
        mismatch_positions=(1,),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
        ),
    )

    result = _select_result(
        candidates=(edge_candidate, middle_candidate),
        pwm_effective=False,
        ligation_profile="none",
        ligation_awareness_mode="secondary",
    )

    assert result.winner.lexical_key == "middle"


def test_t4_prefers_gt_edge_over_ag_middle_when_pwm_equal() -> None:
    gt_edge = _candidate(
        lexical_key="gt-edge",
        default_strand_preference_count=1,
        mismatch_positions=(0,),
        mutations=(
            MutationChoice(
                junction_offset=0,
                payload_index=4,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
        ),
    )
    ag_middle = _candidate(
        lexical_key="ag-middle",
        default_strand_preference_count=1,
        mismatch_positions=(1,),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="A",
            ),
        ),
    )

    result = _select_result(
        candidates=(ag_middle, gt_edge),
        pwm_effective=False,
        ligation_profile="t4",
        ligation_awareness_mode="secondary",
    )

    assert result.winner.lexical_key == "gt-edge"


def test_t4_prefers_edge_gt_over_middle_gt_when_pwm_equal() -> None:
    edge_gt = _candidate(
        lexical_key="edge-gt",
        default_strand_preference_count=1,
        mismatch_positions=(0,),
        mutations=(
            MutationChoice(
                junction_offset=0,
                payload_index=4,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
        ),
    )
    middle_gt = _candidate(
        lexical_key="middle-gt",
        default_strand_preference_count=1,
        mismatch_positions=(1,),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
        ),
    )

    result = _select_result(
        candidates=(middle_gt, edge_gt),
        pwm_effective=False,
        ligation_profile="t4",
        ligation_awareness_mode="secondary",
    )

    assert result.winner.lexical_key == "edge-gt"


def test_t4_penalizes_double_middle_more_than_edge_plus_middle() -> None:
    double_middle = _candidate(
        lexical_key="double-middle",
        default_strand_preference_count=1,
        mismatch_positions=(1, 2),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
            MutationChoice(
                junction_offset=2,
                payload_index=6,
                mutated_strand="payload",
                native_base="C",
                mutated_base="T",
                opposing_base="G",
            ),
        ),
    )
    edge_plus_middle = _candidate(
        lexical_key="edge-plus-middle",
        default_strand_preference_count=1,
        mismatch_positions=(0, 1),
        mutations=(
            MutationChoice(
                junction_offset=0,
                payload_index=4,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
        ),
    )

    result = _select_result(
        candidates=(double_middle, edge_plus_middle),
        pwm_effective=False,
        ligation_profile="t4",
        ligation_awareness_mode="secondary",
    )

    assert result.winner.lexical_key == "edge-plus-middle"


def test_hlig3_allows_ag_or_gg_as_second_tier() -> None:
    ag_middle = _candidate(
        lexical_key="ag-middle",
        default_strand_preference_count=1,
        mismatch_positions=(1,),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="payload",
                native_base="T",
                mutated_base="A",
                opposing_base="G",
            ),
        ),
    )
    ct_middle = _candidate(
        lexical_key="ct-middle",
        default_strand_preference_count=1,
        mismatch_positions=(1,),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="complement",
                native_base="A",
                mutated_base="C",
                opposing_base="T",
            ),
        ),
    )

    result = _select_result(
        candidates=(ct_middle, ag_middle),
        pwm_effective=False,
        ligation_profile="hlig3",
        ligation_awareness_mode="secondary",
    )

    assert result.winner.lexical_key == "ag-middle"


@pytest.mark.parametrize(
    ("ligation_profile", "mismatch_class", "expected_tier"),
    [
        ("t7", "AG", 2),
        ("t7", "GG", 2),
        ("t3", "AG", 1),
        ("pbcv1", "GG", 1),
    ],
)
def test_ligation_profile_tier_tables_cover_remaining_profiles(
    ligation_profile: str,
    mismatch_class: str,
    expected_tier: int,
) -> None:
    from dnadesign.cruncher.yiu.ligation_scoring import mismatch_class_tier

    assert mismatch_class_tier(mismatch_class=mismatch_class, ligation_profile=ligation_profile) == expected_tier


def test_pwm_primary_ligation_secondary(monkeypatch: pytest.MonkeyPatch) -> None:
    better_pwm_but_biologically_worse = _candidate(
        lexical_key="better-pwm",
        default_strand_preference_count=0,
        mismatch_positions=(1,),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="complement",
                native_base="A",
                mutated_base="C",
                opposing_base="T",
            ),
        ),
    )
    better_ligation_but_worse_pwm = _candidate(
        lexical_key="better-ligation",
        default_strand_preference_count=1,
        mismatch_positions=(0,),
        mutations=(
            MutationChoice(
                junction_offset=0,
                payload_index=4,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
        ),
    )

    def _stub_score_candidate(*, candidate, reference_payload_sequence, reference_complement_sequence, scorable_motifs):
        _ = (reference_payload_sequence, reference_complement_sequence, scorable_motifs)
        if candidate.lexical_key == "better-pwm":
            return CandidateScore(worst_loss=0.5, total_loss=0.5)
        return CandidateScore(worst_loss=1.0, total_loss=1.0)

    monkeypatch.setattr(yiu_optimizer_module, "score_candidate", _stub_score_candidate)
    result = _select_result(
        candidates=(better_ligation_but_worse_pwm, better_pwm_but_biologically_worse),
        pwm_effective=True,
        ligation_profile="t4",
        ligation_awareness_mode="secondary",
    )

    assert result.winner.lexical_key == "better-pwm"


def test_pwm_tolerance_then_ligation_prefers_better_ligation_within_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    better_pwm_but_biologically_worse = _candidate(
        lexical_key="better-pwm",
        default_strand_preference_count=0,
        mismatch_positions=(1,),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="complement",
                native_base="A",
                mutated_base="C",
                opposing_base="T",
            ),
        ),
    )
    better_ligation_but_slightly_worse_pwm = _candidate(
        lexical_key="better-ligation",
        default_strand_preference_count=1,
        mismatch_positions=(0,),
        mutations=(
            MutationChoice(
                junction_offset=0,
                payload_index=4,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
        ),
    )

    def _stub_score_candidate(*, candidate, reference_payload_sequence, reference_complement_sequence, scorable_motifs):
        _ = (reference_payload_sequence, reference_complement_sequence, scorable_motifs)
        if candidate.lexical_key == "better-pwm":
            return CandidateScore(worst_loss=0.5, total_loss=0.5)
        return CandidateScore(worst_loss=0.7, total_loss=0.7)

    monkeypatch.setattr(yiu_optimizer_module, "score_candidate", _stub_score_candidate)
    result = _select_result(
        candidates=(better_ligation_but_slightly_worse_pwm, better_pwm_but_biologically_worse),
        pwm_effective=True,
        ligation_profile="t4",
        ligation_awareness_mode="secondary",
        ligation_selection_mode="pwm_tolerance_then_ligation",
        pwm_worst_loss_tolerance=0.25,
        pwm_total_loss_tolerance=0.25,
    )

    assert result.winner.lexical_key == "better-ligation"


def test_secondary_mode_does_not_force_tnna_penalty_without_bad_pattern_heuristics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tnna_candidate = _candidate(
        lexical_key="a-tnna",
        default_strand_preference_count=1,
        mismatch_positions=(3,),
        mutations=(
            MutationChoice(
                junction_offset=3,
                payload_index=7,
                mutated_strand="complement",
                native_base="G",
                mutated_base="T",
                opposing_base="C",
            ),
        ),
    )
    plain_candidate = _candidate(
        lexical_key="z-plain",
        default_strand_preference_count=1,
        mismatch_positions=(0,),
        mutations=(
            MutationChoice(
                junction_offset=0,
                payload_index=4,
                mutated_strand="complement",
                native_base="A",
                mutated_base="C",
                opposing_base="T",
            ),
        ),
    )

    def _stub_score_candidate(*, candidate, reference_payload_sequence, reference_complement_sequence, scorable_motifs):
        _ = (candidate, reference_payload_sequence, reference_complement_sequence, scorable_motifs)
        return CandidateScore(worst_loss=0.0, total_loss=0.0)

    monkeypatch.setattr(yiu_optimizer_module, "score_candidate", _stub_score_candidate)
    result = _select_result(
        candidates=(plain_candidate, tnna_candidate),
        pwm_effective=True,
        ligation_profile="t4",
        ligation_awareness_mode="secondary",
        bad_pattern_heuristics=False,
    )

    assert result.winner.lexical_key == "a-tnna"
    assert result.chosen_ligation_key is not None
    assert result.chosen_ligation_key.bad_pattern_penalty == 0


def test_hard_ligation_filter_demotes_better_pwm_candidate_when_it_violates_ligation_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    better_pwm_but_weak_ligation = _candidate(
        lexical_key="better-pwm",
        default_strand_preference_count=0,
        mismatch_positions=(1,),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="payload",
                native_base="T",
                mutated_base="A",
                opposing_base="A",
            ),
        ),
    )
    admissible_gt_edge = _candidate(
        lexical_key="gt-edge",
        default_strand_preference_count=1,
        mismatch_positions=(0,),
        mutations=(
            MutationChoice(
                junction_offset=0,
                payload_index=4,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
        ),
    )

    def _stub_score_candidate(*, candidate, reference_payload_sequence, reference_complement_sequence, scorable_motifs):
        _ = (reference_payload_sequence, reference_complement_sequence, scorable_motifs)
        if candidate.lexical_key == "better-pwm":
            return CandidateScore(worst_loss=0.1, total_loss=0.1)
        return CandidateScore(worst_loss=0.8, total_loss=0.8)

    monkeypatch.setattr(yiu_optimizer_module, "score_candidate", _stub_score_candidate)
    result = _select_result(
        candidates=(better_pwm_but_weak_ligation, admissible_gt_edge),
        pwm_effective=True,
        ligation_profile="t4",
        ligation_awareness_mode="secondary",
        ligation_selection_mode="hard_ligation_filter",
        max_worst_mismatch_class_tier=0,
        max_middle_mismatch_count=1,
        allow_double_middle=False,
        allow_tnna_like_overhangs=False,
    )

    assert result.winner.lexical_key == "gt-edge"


def test_default_strand_preference_remains_late() -> None:
    preferred = _candidate(lexical_key="z", default_strand_preference_count=1)
    nonpreferred = _candidate(lexical_key="a", default_strand_preference_count=0)

    result = _select_result(
        candidates=(nonpreferred, preferred),
        pwm_effective=False,
        ligation_profile="t4",
        ligation_awareness_mode="secondary",
    )

    assert result.winner.lexical_key == "z"


def test_select_best_candidate_without_pwm_prefers_default_strand_then_lexical_order() -> None:
    preferred = _candidate(lexical_key="z", default_strand_preference_count=1)
    nonpreferred = _candidate(lexical_key="a", default_strand_preference_count=0)

    result = _select_result(candidates=(nonpreferred, preferred), pwm_effective=False)

    assert result.winner.lexical_key == "z"

    lexical_a = _candidate(lexical_key="a", default_strand_preference_count=1)
    lexical_b = _candidate(lexical_key="b", default_strand_preference_count=1)
    lexical_result = _select_result(candidates=(lexical_b, lexical_a), pwm_effective=False)
    assert lexical_result.winner.lexical_key == "a"


def test_load_yiu_spec_rejects_noncanonical_secondary_ladder(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "legacy_secondary.yiu.yaml"
    payload = _user_sequence_spec()
    payload["optimization"]["pwm"]["objective"]["secondary"] = [
        "total_loss",
        "midpoint_proximity",
        "body_length_balance",
        "terminal_position_avoidance",
        "default_strand_preference",
        "lexical_stability",
    ]
    _write_yaml(spec_path, payload)

    with pytest.raises(ValueError, match="canonical Yiu v4 ladder order"):
        load_yiu_spec(spec_path)


def test_mismatch_class_is_orientation_independent() -> None:
    from dnadesign.cruncher.yiu.ligation_scoring import build_candidate_ligation_score

    payload_mutated = _candidate(
        lexical_key="payload-mutated",
        default_strand_preference_count=1,
        mismatch_positions=(1,),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="payload",
                native_base="T",
                mutated_base="G",
                opposing_base="T",
            ),
        ),
    )
    complement_mutated = _candidate(
        lexical_key="complement-mutated",
        default_strand_preference_count=1,
        mismatch_positions=(1,),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
        ),
    )

    payload_score = build_candidate_ligation_score(
        candidate=payload_mutated,
        ligation_profile="t4",
        bad_pattern_heuristics=False,
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
    )
    complement_score = build_candidate_ligation_score(
        candidate=complement_mutated,
        ligation_profile="t4",
        bad_pattern_heuristics=False,
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
    )

    assert payload_score.mismatch_rationales[0].canonical_mismatch_class == "GT"
    assert complement_score.mismatch_rationales[0].canonical_mismatch_class == "GT"
    assert payload_score.key == complement_score.key


def test_bad_pattern_heuristics_can_penalize_tnna_like_overhangs() -> None:
    from dnadesign.cruncher.yiu.ligation_scoring import build_candidate_ligation_score

    tnna_candidate = _candidate(
        lexical_key="tnna-like",
        default_strand_preference_count=1,
        mismatch_positions=(3,),
        mutations=(
            MutationChoice(
                junction_offset=3,
                payload_index=7,
                mutated_strand="complement",
                native_base="G",
                mutated_base="T",
                opposing_base="C",
            ),
        ),
    )

    disabled_score = build_candidate_ligation_score(
        candidate=tnna_candidate,
        ligation_profile="t4",
        bad_pattern_heuristics=False,
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
    )
    enabled_score = build_candidate_ligation_score(
        candidate=tnna_candidate,
        ligation_profile="t4",
        bad_pattern_heuristics=True,
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
    )

    assert disabled_score.chosen_key.bad_pattern_penalty == 0
    assert enabled_score.chosen_key.bad_pattern_penalty == 1


def test_hard_ligation_filter_raises_relaxation_hint_when_it_eliminates_the_pool() -> None:
    middle_only_double_mismatch = _candidate(
        lexical_key="middle-only",
        default_strand_preference_count=1,
        mismatch_positions=(1, 2),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="complement",
                native_base="A",
                mutated_base="G",
                opposing_base="T",
            ),
            MutationChoice(
                junction_offset=2,
                payload_index=6,
                mutated_strand="payload",
                native_base="C",
                mutated_base="T",
                opposing_base="G",
            ),
        ),
    )

    with pytest.raises(NoFeasiblePlanError, match="hard_ligation_filter"):
        _select_result(
            candidates=(middle_only_double_mismatch,),
            pwm_effective=False,
            ligation_profile="t4",
            ligation_awareness_mode="secondary",
            ligation_selection_mode="hard_ligation_filter",
            max_worst_mismatch_class_tier=0,
            max_middle_mismatch_count=0,
            allow_double_middle=False,
            allow_tnna_like_overhangs=False,
        )


def test_hard_ligation_filter_relaxation_hint_uses_smallest_relevant_thresholds() -> None:
    tnna_edge = _candidate(
        lexical_key="tnna-edge",
        default_strand_preference_count=1,
        mismatch_positions=(3,),
        mutations=(
            MutationChoice(
                junction_offset=3,
                payload_index=7,
                mutated_strand="complement",
                native_base="G",
                mutated_base="T",
                opposing_base="C",
            ),
        ),
    )
    middle_only = _candidate(
        lexical_key="middle-only",
        default_strand_preference_count=1,
        mismatch_positions=(1,),
        mutations=(
            MutationChoice(
                junction_offset=1,
                payload_index=5,
                mutated_strand="complement",
                native_base="A",
                mutated_base="C",
                opposing_base="T",
            ),
        ),
    )

    with pytest.raises(NoFeasiblePlanError, match="hard_ligation_filter") as exc_info:
        _select_result(
            candidates=(tnna_edge, middle_only),
            pwm_effective=False,
            ligation_profile="t4",
            ligation_awareness_mode="secondary",
            ligation_selection_mode="hard_ligation_filter",
            max_worst_mismatch_class_tier=3,
            max_middle_mismatch_count=0,
            allow_double_middle=False,
            allow_tnna_like_overhangs=False,
        )

    message = str(exc_info.value)
    assert "max_middle_mismatch_count=1" in message
    assert "allow_tnna_like_overhangs=true" in message
    assert "max_middle_mismatch_count=0" not in message


def test_render_yiu_spec_publishes_v4_bundle_and_payload_visual_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    bundle_dir, report = render_yiu_spec(spec_path)

    assert report.contract == "split_yiu_payload_rendering_v4"
    manifest = _load_json(bundle_dir / "bundle_manifest.json")
    bundle_summary = _load_json(bundle_dir / "bundle_summary.json")
    normalized = _load_json(bundle_dir / "normalized_payload.json")
    inventory = _load_json(bundle_dir / "visual_inventory.json")
    payload_view = _load_json(bundle_dir / "payload_view.json")
    split_rows = _load_jsonl(bundle_dir / "split_payload_view.jsonl")
    assembled_view = _load_json(bundle_dir / "assembled_payload_view.json")

    assert manifest["bundle_contract"] == "split_yiu_payload_bundle_v4"
    assert manifest["published_plot_artifact_path"] == "outputs/plot__demo_payload__payload_views.pdf"
    assert bundle_summary["summary_contract"] == "yiu_bundle_summary_v3"
    normalized_model = NormalizedPayload.model_validate(normalized)
    canonical_model = normalized_model.model_copy(
        update={
            "selected_payload_sequence": normalized_model.reference_payload_sequence,
            "selected_complement_sequence": normalized_model.reference_complement_sequence,
        }
    )
    canonical_fragments = {
        fragment.fragment_side: fragment for fragment in build_split_fragment_display_specs(canonical_model)
    }
    selected_fragments = {
        fragment.fragment_side: fragment for fragment in build_split_fragment_display_specs(normalized_model)
    }
    payload_view_summary = bundle_summary["sequence_summary"]["views"]["payload"]
    split_left_view_summary = bundle_summary["sequence_summary"]["views"]["split_left"]
    split_right_view_summary = bundle_summary["sequence_summary"]["views"]["split_right"]
    assembled_view_summary = bundle_summary["sequence_summary"]["views"]["assembled"]
    assert payload_view_summary["canonical"]["top_strand_5to3"] == normalized["reference_payload_sequence"]
    assert payload_view_summary["canonical"]["bottom_strand_5to3"] == normalized["reference_complement_sequence"][::-1]
    assert payload_view_summary["mismatch_present"]["top_strand_5to3"] == normalized["selected_payload_sequence"]
    assert (
        payload_view_summary["mismatch_present"]["bottom_strand_5to3"] == manifest["selected_complement_sequence"][::-1]
    )
    assert (
        bundle_summary["sequence_summary"]["overhang_5to3"]["canonical_sequence_5to3"]
        == normalized["reference_complement_sequence"][TOY_JUNCTION_START:TOY_JUNCTION_END][::-1]
    )
    assert (
        bundle_summary["sequence_summary"]["overhang_5to3"]["mismatch_present_sequence_5to3"]
        == manifest["selected_complement_sequence"][TOY_JUNCTION_START:TOY_JUNCTION_END][::-1]
    )
    assert (
        split_left_view_summary["canonical"]["top_strand_5to3"]
        == canonical_fragments["left"].retained_primary_sequence_5to3
    )
    assert (
        split_left_view_summary["canonical"]["bottom_strand_5to3"]
        == canonical_fragments["left"].retained_complement_sequence_3to5[::-1]
    )
    assert (
        split_left_view_summary["mismatch_present"]["top_strand_5to3"]
        == selected_fragments["left"].retained_primary_sequence_5to3
    )
    assert (
        split_left_view_summary["mismatch_present"]["bottom_strand_5to3"]
        == selected_fragments["left"].retained_complement_sequence_3to5[::-1]
    )
    assert (
        split_right_view_summary["canonical"]["top_strand_5to3"]
        == canonical_fragments["right"].retained_primary_sequence_5to3
    )
    assert (
        split_right_view_summary["canonical"]["bottom_strand_5to3"]
        == canonical_fragments["right"].retained_complement_sequence_3to5[::-1]
    )
    assert (
        split_right_view_summary["mismatch_present"]["top_strand_5to3"]
        == selected_fragments["right"].retained_primary_sequence_5to3
    )
    assert (
        split_right_view_summary["mismatch_present"]["bottom_strand_5to3"]
        == selected_fragments["right"].retained_complement_sequence_3to5[::-1]
    )
    assert assembled_view_summary == payload_view_summary
    assert bundle_summary["mismatch_notation"] == compact_mismatch_notation_groups(bundle_summary["mismatches"])
    assert normalized["contract"] == "yiu_normalized_payload_v5"
    assert "published_artifacts" not in normalized
    assert inventory["pwm_effective"] is False
    assert inventory["published_plot_artifact_path"] == "outputs/plot__demo_payload__payload_views.pdf"
    assert payload_view["contract_kind"] == "yiu_payload_visual_v1"
    assert payload_view["motif_layers"] == []
    assert len(split_rows) == 2
    assert assembled_view["contract_kind"] == "sequence_evidence_map_v1"


def test_render_yiu_spec_bundle_summary_reports_ligation_rationale(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "ligation_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="ligation_payload",
            candidate_positions=[0, 1],
            ligation_profile="t4",
            ligation_awareness_mode="secondary",
        ),
    )

    bundle_dir, _report = render_yiu_spec(spec_path)
    bundle_summary = _load_json(bundle_dir / "bundle_summary.json")
    normalized = _load_json(bundle_dir / "normalized_payload.json")

    assert bundle_summary["ligation"]["profile"] == "t4"
    assert bundle_summary["ligation"]["awareness_mode"] == "secondary"
    assert bundle_summary["ligation"]["applied"] is True
    assert "ligation-aware" in bundle_summary["ligation"]["decision_note"]
    assert bundle_summary["ligation"]["chosen_mismatch_classes"]
    assert bundle_summary["ligation"]["position_classes"]
    assert normalized["chosen_ligation_key"]["middle_mismatch_count"] >= 0
    assert normalized["ligation_rationale"]


def test_render_yiu_spec_bundle_summary_reports_ligation_policy_mode_and_filter_counts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "strict_ligation_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="strict_ligation_payload",
            candidate_positions=[0, 1, 2, 3],
            ligation_profile="t4",
            ligation_awareness_mode="secondary",
            ligation_selection_mode="hard_ligation_filter",
            max_worst_mismatch_class_tier=2,
            max_middle_mismatch_count=1,
            allow_double_middle=False,
            allow_tnna_like_overhangs=False,
        ),
    )

    bundle_dir, _report = render_yiu_spec(spec_path)
    bundle_summary = _load_json(bundle_dir / "bundle_summary.json")
    normalized = _load_json(bundle_dir / "normalized_payload.json")

    assert bundle_summary["ligation"]["selection_mode"] == "hard_ligation_filter"
    assert bundle_summary["ligation"]["filtered_candidate_count"] > 0
    assert (
        bundle_summary["ligation"]["candidate_count_before_filter"]
        > bundle_summary["ligation"]["candidate_count_after_filter"]
    )
    assert normalized["optimization_decision"]["ligation_policy"]["selection_mode"] == "hard_ligation_filter"
    assert normalized["optimization_decision"]["ligation_policy"]["filter_applied"] is True


def test_load_yiu_spec_normalizes_hard_filter_alias(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "hard_filter_alias.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="hard_filter_alias",
            candidate_positions=[0, 1, 2, 3],
            ligation_profile="t4",
            ligation_selection_mode="hard_filter",
        ),
    )

    spec, _resolved_spec_path, _workspace_root = load_yiu_spec(spec_path)

    assert spec.optimization.mismatches.ligation_selection_mode == "hard_ligation_filter"


def test_render_yiu_spec_bundle_summary_marks_profile_none_as_legacy_mode(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "legacy_ligation_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="legacy_ligation_payload",
            candidate_positions=[0, 1, 2, 3],
            ligation_profile="none",
            ligation_awareness_mode="secondary",
        ),
    )

    bundle_dir, _report = render_yiu_spec(spec_path)
    bundle_summary = _load_json(bundle_dir / "bundle_summary.json")

    assert bundle_summary["ligation"]["profile"] == "none"
    assert bundle_summary["ligation"]["awareness_mode"] == "secondary"
    assert bundle_summary["ligation"]["state"] == "legacy"
    assert bundle_summary["ligation"]["edge_comparison_available"] is False
    assert "legacy mode" in bundle_summary["ligation"]["decision_note"].lower()


def test_render_yiu_spec_bundle_summary_marks_middle_only_pool_as_edge_blind(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "middle_only_ligation_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="middle_only_ligation_payload",
            candidate_positions=[1, 2],
            ligation_profile="t4",
            ligation_awareness_mode="secondary",
        ),
    )

    bundle_dir, _report = render_yiu_spec(spec_path)
    bundle_summary = _load_json(bundle_dir / "bundle_summary.json")

    assert bundle_summary["ligation"]["profile"] == "t4"
    assert bundle_summary["ligation"]["state"] == "edge_blind"
    assert bundle_summary["ligation"]["candidate_positions"] == [1, 2]
    assert bundle_summary["ligation"]["edge_comparison_available"] is False
    assert bundle_summary["ligation"]["position_classes"] == ["middle"]
    assert "candidate_positions excludes 0/3" in bundle_summary["ligation"]["decision_note"]


def test_normalize_payload_caps_optimizer_trace_at_bounded_sample_size(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "dense_trace_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="dense_trace_payload",
            mismatch_count=2,
            candidate_positions=[0, 1, 2, 3],
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    assert len(normalized.optimization_decision.trace) == 32
    assert normalized.optimization_decision.trace[0]["junction_start"] == 4


def test_normalize_payload_exposes_optimizer_trace_truncation_metadata(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "dense_trace_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="dense_trace_payload",
            mismatch_count=2,
            candidate_positions=[0, 1, 2, 3],
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)
    decision = normalized.optimization_decision.model_dump(mode="json")

    assert decision["trace_sample"] == {
        "sample_limit": 32,
        "sampled_count": 32,
        "candidate_count": decision["candidate_count"],
        "truncated": True,
    }


def test_render_yiu_spec_outcome_uses_shared_bundle_surface(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    outcome = render_yiu_spec_outcome(spec_path)

    assert outcome.bundle_dir == str((workspace / "outputs" / "demo_payload").resolve())
    assert outcome.outputs_root == str((workspace / "outputs").resolve())
    assert outcome.composite_render_artifact_path == str(
        (workspace / "outputs" / "demo_payload" / "payload_views.pdf").resolve()
    )
    assert outcome.published_plot_artifact_path == str(
        (workspace / "outputs" / "plot__demo_payload__payload_views.pdf").resolve()
    )
    assert outcome.bundle_summary_path == str(
        (workspace / "outputs" / "demo_payload" / "bundle_summary.json").resolve()
    )
    assert outcome.report.spec_name == "demo_payload"


def test_render_yiu_spec_with_pwm_effective_adds_payload_motif_layers_and_show_roundtrips(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "pwm_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="pwm_payload",
            mismatch_count=2,
            candidate_positions=[0, 1, 2, 3],
            ligation_profile="t4",
            pwm_mode="require",
            pwm_source={"kind": "inline", "inline_context": _inline_pwm_context()},
        ),
    )

    bundle_dir, report = render_yiu_spec(spec_path, emit_renders=True)
    payload_view = _load_json(bundle_dir / "payload_view.json")
    show_payload = show_yiu_bundle(bundle_dir)

    assert isinstance(show_payload, YiuShowOutcome)
    assert report.pwm_effective is True
    assert len(payload_view["motif_layers"]) == 2
    assert (bundle_dir / "payload_views.pdf").exists()
    assert (workspace / "outputs" / "plot__pwm_payload__payload_views.pdf").exists()
    assert show_payload.pwm_effective is True
    assert show_payload.motif_context.effective is True
    assert show_payload.bundle_summary.pwm.motif_count == 2
    assert show_payload.bundle_summary.ligation.profile == "t4"
    assert show_payload.bundle_summary.ligation.applied is True
    assert show_payload.bundle_summary.ligation.chosen_mismatch_classes
    assert show_payload.bundle_summary.ligation.position_classes
    assert show_payload.bundle_summary.ligation.decision_note == "PWM preserved first, ligation-aware tie-break applied"
    assert show_payload.integrity.status == "ok"
    assert show_payload.published_plot_artifact_path == str(
        (workspace / "outputs" / "plot__pwm_payload__payload_views.pdf").resolve()
    )


def test_payload_view_uses_sample_inspired_pwm_style() -> None:
    overrides = build_yiu_style_overrides("payload")

    assert overrides["legend"] is False
    assert overrides["connectors"] is True
    assert overrides["sequence"]["bold_consensus_bases"] is True
    assert overrides["motif_logo"]["letter_coloring"]["mode"] == "match_window_seq"
    assert overrides["motif_logo"]["letter_coloring"]["observed_color_source"] == "feature_fill"


def test_operator_strip_views_scale_up_for_legibility() -> None:
    payload_overrides = build_yiu_style_overrides("payload")
    split_overrides = build_yiu_style_overrides("split_payload")
    assembled_overrides = build_yiu_style_overrides("assembled_payload")

    assert split_overrides["overlay_align"] == "center"
    assert assembled_overrides["overlay_align"] == "center"
    assert payload_overrides["overlay_title_color"] == "#111827"
    assert split_overrides["overlay_title_color"] == "#111827"
    assert assembled_overrides["overlay_title_color"] == "#111827"
    assert split_overrides["figure_scale"] == payload_overrides["figure_scale"]
    assert assembled_overrides["figure_scale"] == payload_overrides["figure_scale"]
    assert split_overrides["font_size_seq"] == payload_overrides["font_size_seq"]
    assert assembled_overrides["font_size_seq"] == payload_overrides["font_size_seq"]
    assert split_overrides["font_size_label"] == payload_overrides["font_size_label"]
    assert assembled_overrides["font_size_label"] == payload_overrides["font_size_label"]
    assert split_overrides["connectors"] is True
    assert assembled_overrides["connectors"] is True
    assert payload_overrides["overlay_title_gap_reduction_px"] > 0
    assert split_overrides["overlay_title_gap_reduction_px"] > payload_overrides["overlay_title_gap_reduction_px"]
    assert assembled_overrides["overlay_title_gap_reduction_px"] > payload_overrides["overlay_title_gap_reduction_px"]


def test_operator_strip_views_inherit_bench_strip_foundation() -> None:
    payload_overrides = build_yiu_style_overrides("payload")
    split_overrides = build_yiu_style_overrides("split_payload")
    assembled_overrides = build_yiu_style_overrides("assembled_payload")

    assert split_overrides["sequence"]["bold_consensus_bases"] is True
    assert assembled_overrides["sequence"]["bold_consensus_bases"] is True
    assert split_overrides["sequence"]["non_consensus_color"] == payload_overrides["sequence"]["non_consensus_color"]
    assert assembled_overrides["kmer"]["box_height_cells"] == split_overrides["kmer"]["box_height_cells"]


def test_yiu_style_profiles_return_defensive_copies() -> None:
    profile = get_yiu_style_profile("payload")
    profile.style_overrides["sequence"]["bold_consensus_bases"] = False

    refreshed = get_yiu_style_profile("payload")

    assert refreshed.system_name == "bench_strip"
    assert refreshed.direction_name == "evidence_ribbon"
    assert refreshed.style_overrides["sequence"]["bold_consensus_bases"] is True


def test_yiu_style_overrides_fail_fast_for_unknown_view_ids() -> None:
    with pytest.raises(ValueError, match="unsupported YIU view id"):
        build_yiu_style_overrides("bogus")


def test_split_payload_view_metadata_preserves_sticky_end_and_ghost_context_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)
    right_fragment = max(build_split_fragment_display_specs(normalized), key=lambda item: item.panel_order)
    right_row = max(build_split_payload_view_rows(normalized), key=lambda item: item["state_id"])
    assert right_fragment.ghost_excised_context is not None

    meta = build_split_payload_row_meta(right_fragment, normalized)
    sticky_end_span = right_fragment.sticky_end_display_span.model_dump(mode="json")
    assert meta["fragment_side"] == "right"
    assert meta["payload_body_sequence_5to3"] == normalized.selected_payload_sequence[normalized.junction.end :]
    assert meta["display_payload_body_sequence_5to3"] == right_fragment.display_payload_body_sequence_5to3
    assert meta["selected_sticky_end_sequence_5to3"] == right_fragment.selected_sticky_end_sequence_5to3
    assert meta["canonical_sticky_end_sequence_5to3"] == right_fragment.canonical_sticky_end_sequence_5to3
    assert meta["sticky_end_display_span"] == sticky_end_span
    assert meta["payload_junction_window"] == right_fragment.payload_junction_window.model_dump(mode="json")
    assert meta["base_highlights"] == {"primary": [], "complement": []}
    assert meta["base_highlight_color"] == "#B91C1C"
    assert meta["connector_hidden_indices"] == []
    assert meta["connector_cross_indices"] == []
    assert meta["connector_overhang_spans"] == [sticky_end_span]
    assert meta["span_backdrops"] == [
        {
            "start": sticky_end_span["start"],
            "end": sticky_end_span["end"],
            "coordinate_space": sticky_end_span["coordinate_space"],
            "fill": "#BFDBFE",
            "alpha": 0.3,
            "corner_radius": 8.0,
            "cover_rows": "both",
        }
    ]
    assert meta["ghost_excised_context"] == right_fragment.ghost_excised_context.model_dump(mode="json")
    assert meta["dim_base_indices"] == {
        "primary": list(right_fragment.ghost_excised_context.primary_indices),
        "complement": list(right_fragment.ghost_excised_context.complement_indices),
    }
    assert right_row["boundaries"] == [
        {
            "boundary_id": "junction_start",
            "row_id": "primary",
            "boundary": sticky_end_span["start"],
            "boundary_kind": "ligation_junction",
            "display_label": "Junction start",
            "short_label": "",
        },
        {
            "boundary_id": "junction_end",
            "row_id": "complement",
            "boundary": sticky_end_span["end"],
            "boundary_kind": "ligation_junction",
            "display_label": "Junction end",
            "short_label": "",
        },
    ]


def test_split_payload_view_metadata_partitions_mixed_strand_mismatches_into_split_rows() -> None:
    normalized = NormalizedPayload.model_validate(
        {
            "name": "mixed_split_payload",
            "input_kind": "user_sequence",
            "reference_payload_sequence": "CTGTATTTATATACAG",
            "reference_complement_sequence": "GACATAAATATATGTC",
            "selected_payload_sequence": "CTGTATAAATATACAG",
            "selected_complement_sequence": "GACATAAATATATGTC",
            "source_provenance": {},
            "ligation_profile": "none",
            "ligation_awareness_mode": "disabled",
            "bad_pattern_heuristics": False,
            "chosen_ligation_key": None,
            "ligation_rationale": [],
            "junction": {
                "start": 5,
                "end": 9,
                "offsets": [0, 1, 2, 3],
                "mode": "optimize",
                "left_body_length": 5,
                "right_body_length": 7,
            },
            "mismatches": [
                {
                    "payload_index": 6,
                    "junction_offset": 1,
                    "mutated_strand": "complement",
                    "native_base": "T",
                    "mutated_base": "A",
                    "opposing_base": "A",
                },
                {
                    "payload_index": 7,
                    "junction_offset": 2,
                    "mutated_strand": "payload",
                    "native_base": "T",
                    "mutated_base": "A",
                    "opposing_base": "A",
                },
            ],
            "motif_context": {
                "requested_mode": "none",
                "effective": False,
                "source_kind": "none",
                "motifs": [],
            },
            "optimization_decision": {
                "candidate_count": 1,
                "objective": {"primary": "maximin", "secondary": []},
                "winner": {
                    "junction_start": 5,
                    "junction_end": 9,
                    "selected_positions": [1, 2],
                    "mutated_strands": ["complement", "payload"],
                    "mutated_bases": ["A", "A"],
                    "worst_loss": 0.0,
                    "total_loss": 0.0,
                    "midpoint_distance": 0,
                    "middle_mismatch_count": 2,
                    "double_middle_flag": True,
                    "default_strand_preference_count": 1,
                    "lexical_key": "mixed-split",
                },
                "trace": [],
            },
        }
    )

    left_fragment, right_fragment = build_split_fragment_display_specs(normalized)
    left_meta = build_split_payload_row_meta(left_fragment, normalized)
    right_meta = build_split_payload_row_meta(right_fragment, normalized)

    assert left_meta["base_highlights"] == {"primary": [9], "complement": []}
    assert right_meta["base_highlights"] == {"primary": [], "complement": [8]}
    assert left_fragment.display_complement_sequence_3to5[8:10] == "AA"
    assert right_fragment.display_complement_sequence_3to5[8:10] == "AA"


def test_assembled_payload_view_metadata_preserves_junction_connector_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec(mismatch_count=2, candidate_positions=[1, 2]))

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)
    assembled_contract = build_assembled_payload_view_contract(normalized)

    meta = build_assembled_payload_view_meta(normalized)
    junction_span = {
        "start": normalized.junction.start,
        "end": normalized.junction.end,
        "coordinate_space": "payload_forward",
    }
    mismatch_indices = [site.payload_index for site in normalized.mismatches]

    assert meta["junction_span"] == junction_span
    assert meta["mismatches"] == [site.model_dump(mode="json") for site in normalized.mismatches]
    assert meta["base_highlights"] == {"primary": [], "complement": mismatch_indices}
    assert meta["base_highlight_color"] == "#B91C1C"
    assert meta["connector_hidden_indices"] == []
    assert meta["connector_cross_indices"] == []
    assert meta["connector_overhang_spans"] == [junction_span]
    assert meta["span_backdrops"] == [
        {
            "start": junction_span["start"],
            "end": junction_span["end"],
            "coordinate_space": junction_span["coordinate_space"],
            "fill": "#BFDBFE",
            "alpha": 0.3,
            "corner_radius": 8.0,
            "cover_rows": "both",
        }
    ]
    assert assembled_contract["boundaries"] == [
        {
            "boundary_id": "junction_start",
            "row_id": "primary",
            "boundary": normalized.junction.start,
            "boundary_kind": "ligation_junction",
            "display_label": "Junction start",
            "short_label": "",
        },
        {
            "boundary_id": "junction_end",
            "row_id": "complement",
            "boundary": normalized.junction.end,
            "boundary_kind": "ligation_junction",
            "display_label": "Junction end",
            "short_label": "",
        },
    ]


def test_payload_view_content_preserves_motif_mismatch_and_meta_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "pwm_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="pwm_payload",
            mismatch_count=2,
            candidate_positions=[1, 2],
            pwm_mode="require",
            pwm_source={"kind": "inline", "inline_context": _inline_pwm_context()},
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    motif_layers = build_payload_motif_layers(normalized)
    mismatch_annotations = build_payload_mismatch_annotations(normalized)
    meta = build_payload_view_meta(normalized)
    payload_view = build_payload_view_contract(normalized)

    assert [layer.motif_instance_id for layer in motif_layers] == [
        motif.motif_instance_id for motif in normalized.motif_context.motifs
    ]
    assert [layer.label for layer in motif_layers] == [
        f"{motif.tf_name} ({motif.reference_strand})" for motif in normalized.motif_context.motifs
    ]
    assert [layer.matrix for layer in motif_layers] == [
        [list(row) for row in motif.probabilities.rows] for motif in normalized.motif_context.motifs
    ]
    assert [entry.model_dump(mode="json") for entry in mismatch_annotations] == [
        entry.model_dump(mode="json") for entry in normalized.mismatches
    ]
    assert meta == {
        "payload_label": normalized.payload_label,
        "site_label": normalized.site_label,
        "row_labels": {},
        "pwm_effective": normalized.motif_context.effective,
        "motif_ids": [motif.motif_instance_id for motif in normalized.motif_context.motifs],
        "span_backdrops": [
            {
                "start": normalized.junction.start,
                "end": normalized.junction.end,
                "coordinate_space": "payload_forward",
                "fill": "#BFDBFE",
                "alpha": 0.3,
                "corner_radius": 8.0,
                "cover_rows": "both",
            }
        ],
    }
    assert payload_view["motif_layers"] == [layer.model_dump(mode="json") for layer in motif_layers]
    assert payload_view["mismatches"] == [entry.model_dump(mode="json") for entry in mismatch_annotations]
    assert payload_view["meta"] == meta


def test_publish_layout_tracks_relative_view_artifacts_and_entries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)
    layout = resolve_payload_bundle_layout(workspace / spec.output.bundle_dir)
    view_entries = build_payload_view_entries(layout=layout, normalized=normalized)

    assert layout.relative_artifact_path(layout.bundle_summary_path) == "bundle_summary.json"
    assert layout.relative_artifact_path(layout.payload_view_path) == "payload_view.json"
    assert layout.relative_artifact_path(layout.split_payload_view_path) == "split_payload_view.jsonl"
    assert layout.relative_artifact_path(layout.assembled_payload_view_path) == "assembled_payload_view.json"
    assert layout.relative_artifact_path(layout.composite_render_path) == "payload_views.pdf"
    assert [entry.view_id for entry in view_entries] == ["payload", "split_payload", "assembled_payload"]
    assert [entry.view_contract_path for entry in view_entries] == [
        "payload_view.json",
        "split_payload_view.jsonl",
        "assembled_payload_view.json",
    ]
    assert [entry.visual_direction for entry in view_entries] == [
        "evidence_ribbon",
        "operator_strip",
        "operator_strip",
    ]
    assert {entry.render_artifact_path for entry in view_entries} == {"payload_views.pdf"}
    assert view_entries[0].motif_layers_required is False


def test_publish_io_writes_bundle_artifacts_at_canonical_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)
    layout = resolve_payload_bundle_layout(workspace / spec.output.bundle_dir)
    payload_contract = build_payload_view_contract(normalized)
    split_payload_rows = build_split_payload_view_rows(normalized)
    assembled_payload_contract = build_assembled_payload_view_contract(normalized)
    normalized_payload_dump = build_normalized_payload_dump(normalized=normalized)
    view_entries = build_payload_view_entries(layout=layout, normalized=normalized)
    inventory = build_payload_visual_inventory(
        spec=spec,
        normalized=normalized,
        layout=layout,
        view_entries=view_entries,
    )
    manifest = build_payload_bundle_manifest(normalized=normalized, inventory=inventory)
    legacy_split_path = layout.bundle_dir / "split_payload_view.json"
    legacy_split_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_split_path.write_text("{}\n", encoding="utf-8")

    write_payload_bundle_views(
        layout=layout,
        payload_contract=payload_contract,
        split_payload_rows=split_payload_rows,
        assembled_payload_contract=assembled_payload_contract,
    )
    write_normalized_payload_dump(layout=layout, normalized_payload_dump=normalized_payload_dump)
    write_payload_bundle_state(layout=layout, manifest=manifest, inventory=inventory)

    assert _load_json(layout.payload_view_path) == payload_contract
    assert _load_jsonl(layout.split_payload_view_path) == split_payload_rows
    assert not legacy_split_path.exists()
    assert _load_json(layout.assembled_payload_view_path) == assembled_payload_contract
    assert _load_json(layout.normalized_payload_path) == normalized_payload_dump
    assert _load_json(layout.manifest_path) == manifest.model_dump(mode="json")
    assert _load_json(layout.inventory_path) == inventory.model_dump(mode="json")


def test_publish_io_writes_debug_render_jobs_from_view_entries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec(emit_render_jobs_debug=True))

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)
    layout = resolve_payload_bundle_layout(workspace / spec.output.bundle_dir)
    view_entries = build_payload_view_entries(layout=layout, normalized=normalized)

    write_debug_render_jobs(layout=layout, view_entries=view_entries)

    for entry in view_entries:
        job_path = layout.render_jobs_dir / f"{entry.view_id}.job.yaml"
        assert job_path.exists()
        expected_job = yaml.safe_load(yaml.safe_dump(build_render_job_payload(entry=entry), sort_keys=False))
        assert yaml.safe_load(job_path.read_text(encoding="utf-8")) == expected_job


def test_publish_inventory_and_manifest_share_bundle_state_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "pwm_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="pwm_payload",
            mismatch_count=2,
            candidate_positions=[1, 2],
            pwm_mode="require",
            pwm_source={"kind": "inline", "inline_context": _inline_pwm_context()},
        ),
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)
    layout = resolve_payload_bundle_layout(workspace / spec.output.bundle_dir)
    view_entries = build_payload_view_entries(layout=layout, normalized=normalized)
    inventory = build_payload_visual_inventory(
        spec=spec,
        normalized=normalized,
        layout=layout,
        view_entries=view_entries,
    )
    manifest = build_payload_bundle_manifest(normalized=normalized, inventory=inventory)
    normalized_dump = build_normalized_payload_dump(normalized=normalized)

    assert inventory.render_status == "not_requested"
    assert inventory.payload_view_requires_motif_layers is True
    assert manifest.render_status == inventory.render_status
    assert manifest.view_contracts == inventory.views
    assert manifest.composite_render_artifact_path == inventory.composite_render_artifact_path == "payload_views.pdf"
    assert manifest.published_plot_artifact_path == inventory.published_plot_artifact_path
    assert manifest.published_plot_artifact_path == "outputs/plot__pwm_payload__payload_views.pdf"
    assert "published_artifacts" not in normalized_dump


def test_bundle_path_contract_rejects_divergent_view_render_targets(tmp_path: Path) -> None:
    inventory = PayloadVisualInventory(
        spec_name="demo_payload",
        input_kind="user_sequence",
        view_count=2,
        render_count=0,
        views=[
            PayloadViewEntry(
                view_id="payload",
                visual_direction="evidence_ribbon",
                contract_kind="yiu_payload_visual_v1",
                input_kind="json",
                view_contract_path="payload_view.json",
                render_artifact_path="payload_views.pdf",
                renderer_kind="nucleotide_evidence_map",
            ),
            PayloadViewEntry(
                view_id="split_payload",
                visual_direction="operator_strip",
                contract_kind="sequence_evidence_map_v1",
                input_kind="jsonl",
                view_contract_path="split_payload_view.jsonl",
                render_artifact_path="split_payload.pdf",
                renderer_kind="sequence_rows",
            ),
        ],
        composite_render_artifact_path="payload_views.pdf",
    )

    with pytest.raises(ValueError, match="diverge"):
        resolve_composite_render_artifact_path(tmp_path, inventory)


def test_render_sample_hit_with_file_backed_pwm_renders_payload_motif_layers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "tetr_monotypic_hit.yiu.yaml"
    pwm_path = workspace / "motifs" / "tetr_monotypic_pwm_context.yaml"
    _write_yaml(pwm_path, _tetr_pwm_context())
    _write_yaml(
        spec_path,
        _sample_hit_spec(
            name="tetr_monotypic_hit",
            sample_name="tetr",
            hit_id="demo_monotypic_tetr_elite_c4c42365d66b",
            payload_sequence="CTCTATATCTGATATAGAG",
            metadata={"tf_name": "tetR", "motif_name": "tetr_demo", "site_label": "tetr_site"},
            junction_mode="explicit_window",
            junction_start=8,
            junction_end=12,
            mismatch_count=2,
            candidate_positions=[1, 2],
            pwm_mode="require",
            pwm_source={"kind": "file", "path": "motifs/tetr_monotypic_pwm_context.yaml"},
        ),
    )

    bundle_dir, report = render_yiu_spec(spec_path, emit_renders=True)
    payload_view = _load_json(bundle_dir / "payload_view.json")
    show_payload = show_yiu_bundle(bundle_dir)

    assert report.pwm_effective is True
    assert len(payload_view["motif_layers"]) == 1
    assert (workspace / "outputs" / "plots" / "plot__yiu__tetr_monotypic_hit__payload_views.pdf").exists()
    assert payload_view["display"]["title"] == "TetR payload with 1 motif site"
    assert payload_view["motif_layers"][0]["tf_name"] == "tetR"
    assert payload_view["motif_layers"][0]["reference_strand"] == "+"
    assert payload_view["motif_layers"][0]["start"] == 0
    assert payload_view["motif_layers"][0]["end"] == 17
    assert payload_view["meta"]["row_labels"] == {}
    for got, expected in zip(payload_view["motif_layers"][0]["matrix"], _canonical_tetr_pwm_rows(), strict=True):
        assert got == pytest.approx(expected)
    assert show_payload.pwm_effective is True
    assert show_payload.motif_context.effective is True
    assert (
        show_payload.bundle_summary.sequence_summary.overhang_5to3.mismatch_present_sequence_5to3
        == show_payload.selected_complement_sequence[8:12][::-1]
    )
    assert show_payload.integrity.status == "ok"
    assert show_payload.published_plot_artifact_path == str(
        (workspace / "outputs" / "plots" / "plot__yiu__tetr_monotypic_hit__payload_views.pdf").resolve()
    )

    split_rows = _load_jsonl(bundle_dir / "split_payload_view.jsonl")
    assert split_rows[0]["meta"]["row_labels"] == {}
    assert split_rows[1]["meta"]["row_labels"] == {}
    assert split_rows[0]["display"]["title"] == "Left split fragment"
    assert split_rows[1]["display"]["title"] == "Right split fragment"
    assert split_rows[0]["meta"]["payload_body_sequence_5to3"] == payload_view["selected_payload_sequence"][:8]
    assert split_rows[0]["meta"]["display_payload_body_sequence_5to3"] == reverse_complement_iupac(
        payload_view["selected_payload_sequence"][:8]
    )
    assert split_rows[1]["meta"]["payload_body_sequence_5to3"] == payload_view["selected_payload_sequence"][12:]
    assert split_rows[1]["meta"]["display_payload_body_sequence_5to3"] == reverse_complement_iupac(
        payload_view["selected_payload_sequence"][12:]
    )
    assert split_rows[0]["boundaries"] == [
        {
            "boundary_id": "junction_start",
            "row_id": "primary",
            "boundary": 7,
            "boundary_kind": "ligation_junction",
            "display_label": "Junction start",
            "short_label": "",
        },
        {
            "boundary_id": "junction_end",
            "row_id": "complement",
            "boundary": 11,
            "boundary_kind": "ligation_junction",
            "display_label": "Junction end",
            "short_label": "",
        },
    ]
    assert split_rows[1]["boundaries"] == split_rows[0]["boundaries"]
    assert split_rows[0]["meta"]["base_highlights"] == {"primary": [8, 9], "complement": []}
    assert split_rows[1]["meta"]["base_highlights"] == {"primary": [], "complement": []}
    assert split_rows[0]["meta"]["base_highlight_color"] == "#B91C1C"
    assert split_rows[1]["meta"]["base_highlight_color"] == "#B91C1C"
    assert split_rows[0]["meta"]["dim_base_indices"] == {
        "primary": list(range(0, 7)),
        "complement": list(range(0, 11)),
    }
    assert split_rows[1]["meta"]["dim_base_indices"] == {
        "primary": list(range(7, 18)),
        "complement": list(range(11, 18)),
    }
    assembled_view = _load_json(bundle_dir / "assembled_payload_view.json")
    assert assembled_view["display"]["title"] == "Reassembled payload"
    assert assembled_view["meta"]["row_labels"] == {}
    assert assembled_view["meta"]["base_highlights"] == {"primary": [], "complement": [9, 10]}
    assert assembled_view["meta"]["base_highlight_color"] == "#B91C1C"
    assert assembled_view["boundaries"] == [
        {
            "boundary_id": "junction_start",
            "row_id": "primary",
            "boundary": 8,
            "boundary_kind": "ligation_junction",
            "display_label": "Junction start",
            "short_label": "",
        },
        {
            "boundary_id": "junction_end",
            "row_id": "complement",
            "boundary": 12,
            "boundary_kind": "ligation_junction",
            "display_label": "Junction end",
            "short_label": "",
        },
    ]


def test_render_sample_hit_with_sample_context_and_overlapping_selected_occurrences_stacks_pwm_layers(
    tmp_path: Path,
) -> None:
    sibling_workspace = tmp_path / "demo_monotypic_baer"
    artifact_path = sibling_workspace / "outputs" / "optimize" / "tables" / "elites.parquet"
    occurrences_path = sibling_workspace / "outputs" / "optimize" / "tables" / "elites_occurrences.parquet"
    _write_sample_parquet(
        artifact_path,
        hit_id="demo_monotypic_baer_elite_001",
        sequence="TTTTTCCCCCAAAA",
    )
    _write_occurrences_parquet(
        occurrences_path,
        rows=[
            {
                "elite_id": "demo_monotypic_baer_elite_001",
                "tf": "baeR",
                "occurrence_rank": 1,
                "start": 0,
                "end": 11,
                "strand": "+",
                "selected": True,
            },
            {
                "elite_id": "demo_monotypic_baer_elite_001",
                "tf": "baeR",
                "occurrence_rank": 2,
                "start": 2,
                "end": 13,
                "strand": "+",
                "selected": True,
            },
            {
                "elite_id": "demo_monotypic_baer_elite_001",
                "tf": "baeR",
                "occurrence_rank": 3,
                "start": 3,
                "end": 14,
                "strand": "+",
                "selected": True,
            },
        ],
    )
    _write_sample_pwm_config(
        sibling_workspace / "outputs" / "meta" / "config_used.yaml",
        tf_rows={"baeR": _rounded_pwm_rows("T", width=11)},
    )

    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "baer_monotypic_hit.yiu.yaml"
    _write_yaml(
        spec_path,
        _sample_hit_spec(
            name="baer_monotypic_hit",
            hit_id="demo_monotypic_baer_elite_001",
            sample_name="baer_monotypic",
            source_artifact_path=str(artifact_path.resolve()),
            pwm_mode="require",
            pwm_source={"kind": "sample_context"},
            mismatch_count=2,
            candidate_positions=[1, 2],
        ),
    )

    bundle_dir, report = render_yiu_spec(spec_path, emit_renders=True)
    payload_view = _load_json(bundle_dir / "payload_view.json")
    show_payload = show_yiu_bundle(bundle_dir)

    assert report.pwm_effective is True
    assert len(payload_view["motif_layers"]) == 3
    assert payload_view["display"]["title"] == "BaeR payload with 3 motif sites"
    assert payload_view["meta"]["row_labels"] == {}
    assert [layer["start"] for layer in payload_view["motif_layers"]] == [0, 2, 3]
    assert (bundle_dir / "payload_views.pdf").exists()
    assert show_payload.integrity.status == "ok"


def test_checked_in_tetr_pwm_context_preserves_full_information_content() -> None:
    motif = _tetr_pwm_context()["motifs"][0]
    rows = motif["probabilities"]["rows"]
    info_bits: list[float] = []
    for row in rows:
        entropy = 0.0
        for prob in row:
            if prob > 0:
                entropy += -prob * math.log2(prob)
        info_bits.append(max(0.0, 2.0 - entropy))

    assert motif["start"] == 0
    assert motif["end"] == 17
    for got, expected in zip(rows, _canonical_tetr_pwm_rows(), strict=True):
        assert got == pytest.approx(expected)
    assert max(info_bits) - min(info_bits) > 0.35


def test_checked_in_tetr_direct_user_sequence_spec_renders_without_sample_handoff_or_pwm(tmp_path: Path) -> None:
    source_workspace = Path(__file__).resolve().parents[2] / "workspaces" / "demo_monotypic_tetr"
    workspace = tmp_path / "demo_monotypic_tetr"
    shutil.copytree(
        source_workspace,
        workspace,
        ignore=shutil.ignore_patterns(".DS_Store", "outputs", ".cruncher"),
    )

    spec_path = workspace / "configs" / "yiu" / "tetr_teto2_wt_direct.yiu.yaml"

    bundle_dir, report = render_yiu_spec(spec_path, emit_renders=True)
    payload_view = _load_json(bundle_dir / "payload_view.json")
    show_payload = show_yiu_bundle(bundle_dir)

    assert report.input_kind == "user_sequence"
    assert report.pwm_effective is False
    assert report.payload_length == 19
    assert payload_view["reference_payload_sequence"] == "TCCCTATCAGTGATAGAGA"
    assert len(payload_view["selected_payload_sequence"]) == 19
    assert payload_view["motif_layers"] == []
    assert payload_view["meta"]["pwm_effective"] is False
    assert (bundle_dir / "payload_views.pdf").exists()
    assert show_payload.input_kind == "user_sequence"
    assert show_payload.pwm_effective is False
    assert show_payload.published_plot_artifact_path is None
    assert show_payload.integrity.status == "ok"


def test_checked_in_yiu_demo_bundles_roundtrip_show() -> None:
    workspaces_root = Path(__file__).resolve().parents[2] / "workspaces"
    current_normalized_contract = NormalizedPayload.model_fields["contract"].default
    bundle_dirs = [
        workspaces_root / "demo_yiu_payload" / "outputs" / "example_payload",
        workspaces_root / "demo_monotypic_baer" / "outputs" / "plots" / "yiu__baer_monotypic_hit",
        workspaces_root / "demo_monotypic_cpxr" / "outputs" / "plots" / "yiu__cpxr_monotypic_hit",
        workspaces_root / "demo_monotypic_lexa" / "outputs" / "plots" / "yiu__lexa_monotypic_hit",
        workspaces_root / "demo_monotypic_soxr" / "outputs" / "plots" / "yiu__soxr_monotypic_hit",
        workspaces_root / "demo_monotypic_soxs" / "outputs" / "plots" / "yiu__soxs_monotypic_hit",
        workspaces_root / "demo_monotypic_tetr" / "outputs" / "plots" / "yiu__tetr_monotypic_hit",
        workspaces_root / "demo_monotypic_tetr" / "outputs" / "plots" / "yiu__tetr_teto2_wt_direct",
    ]
    if any(not bundle_dir.exists() for bundle_dir in bundle_dirs):
        pytest.skip("checked-in demo bundle outputs are not present in this checkout")
    if any(
        json.loads((bundle_dir / "normalized_payload.json").read_text(encoding="utf-8")).get("contract")
        != current_normalized_contract
        for bundle_dir in bundle_dirs
    ):
        pytest.skip("checked-in demo bundle outputs were generated with an older normalized payload contract")

    for bundle_dir in bundle_dirs:
        outcome = show_yiu_bundle(bundle_dir)
        assert outcome.integrity.status == "ok"
        assert outcome.bundle_summary.mismatch_notation


def test_show_yiu_bundle_rejects_payload_view_drift_when_pwm_effective_but_motifs_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "pwm_payload.yiu.yaml"
    _write_yaml(
        spec_path,
        _user_sequence_spec(
            name="pwm_payload",
            mismatch_count=2,
            candidate_positions=[1, 2],
            pwm_mode="require",
            pwm_source={"kind": "inline", "inline_context": _inline_pwm_context()},
        ),
    )

    bundle_dir, _report = render_yiu_spec(spec_path)
    payload_view_path = bundle_dir / "payload_view.json"
    payload_view = _load_json(payload_view_path)
    payload_view["motif_layers"] = []
    payload_view_path.write_text(json.dumps(payload_view, indent=2), encoding="utf-8")

    with pytest.raises(YiuContractError, match="zero motif layers"):
        show_yiu_bundle(bundle_dir)


def test_show_yiu_bundle_rejects_missing_pdf_when_inventory_claims_rendered(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    bundle_dir, _report = render_yiu_spec(spec_path, emit_renders=True)
    pdf_path = bundle_dir / "payload_views.pdf"
    pdf_path.unlink()

    with pytest.raises(YiuContractError, match="rendered outputs that are missing on disk"):
        show_yiu_bundle(bundle_dir)


def test_show_yiu_bundle_rejects_bundle_summary_drift(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    bundle_dir, _report = render_yiu_spec(spec_path)
    summary_path = bundle_dir / "bundle_summary.json"
    bundle_summary = _load_json(summary_path)
    bundle_summary["sequence_summary"]["views"]["payload"]["canonical"]["bottom_strand_5to3"] = "WRONG"
    summary_path.write_text(json.dumps(bundle_summary, indent=2), encoding="utf-8")

    with pytest.raises(YiuContractError, match="bundle_summary.json disagrees"):
        show_yiu_bundle(bundle_dir)


def test_show_yiu_bundle_rejects_stale_pdf_when_inventory_claims_not_requested(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    bundle_dir, _report = render_yiu_spec(spec_path)
    pdf_path = bundle_dir / "payload_views.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% stale artifact\n")

    with pytest.raises(YiuContractError, match="artifacts exist on disk"):
        show_yiu_bundle(bundle_dir)


def test_render_bundle_views_rejects_corrupt_view_registry_before_mutation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    bundle_dir, _report = render_yiu_spec(spec_path)
    inventory_path = bundle_dir / "visual_inventory.json"
    manifest_path = bundle_dir / "bundle_manifest.json"
    inventory = _load_json(inventory_path)
    manifest = _load_json(manifest_path)

    inventory["view_count"] = 1
    inventory["views"] = inventory["views"][:1]
    manifest["view_contracts"] = manifest["view_contracts"][:1]

    inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(YiuContractError, match="canonical YIU view ids"):
        yiu_render_module.render_bundle_views(bundle_dir)

    persisted_inventory = _load_json(inventory_path)
    persisted_manifest = _load_json(manifest_path)
    assert persisted_inventory["render_status"] == "not_requested"
    assert persisted_inventory["render_count"] == 0
    assert persisted_manifest["render_status"] == "not_requested"


def test_render_yiu_spec_emit_renders_marks_bundle_rendered(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    bundle_dir, _report = render_yiu_spec(spec_path, emit_renders=True)
    inventory = _load_json(bundle_dir / "visual_inventory.json")
    manifest = _load_json(bundle_dir / "bundle_manifest.json")

    assert inventory["render_status"] == "rendered"
    assert inventory["render_count"] == 3
    assert manifest["render_status"] == "rendered"
    assert (bundle_dir / "payload_views.pdf").exists()


def test_render_yiu_spec_force_overwrite_preserves_live_bundle_on_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    bundle_dir, _report = render_yiu_spec(spec_path)
    manifest_before = _load_json(bundle_dir / "bundle_manifest.json")

    def _raise_publish_failure(*args, **kwargs):
        raise RuntimeError("synthetic publish failure")

    monkeypatch.setattr(yiu_workflow_render_module, "publish_payload_bundle", _raise_publish_failure)

    with pytest.raises(RuntimeError, match="synthetic publish failure"):
        render_yiu_spec(spec_path, force_overwrite=True)

    assert bundle_dir.exists()
    assert _load_json(bundle_dir / "bundle_manifest.json") == manifest_before
    assert list((workspace / "outputs").glob(".demo_payload.staging.*")) == []


def test_render_yiu_spec_force_overwrite_without_emit_renders_cleans_stale_published_plot(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    bundle_dir, _report = render_yiu_spec(spec_path, emit_renders=True)
    published_plot_path = workspace / "outputs" / "plot__demo_payload__payload_views.pdf"

    assert published_plot_path.exists()
    assert (bundle_dir / "payload_views.pdf").exists()

    overwritten_bundle_dir, _report = render_yiu_spec(spec_path, force_overwrite=True, emit_renders=False)
    inventory = _load_json(overwritten_bundle_dir / "visual_inventory.json")
    manifest = _load_json(overwritten_bundle_dir / "bundle_manifest.json")

    assert overwritten_bundle_dir == bundle_dir
    assert inventory["render_status"] == "not_requested"
    assert inventory["render_count"] == 0
    assert manifest["render_status"] == "not_requested"
    assert not (bundle_dir / "payload_views.pdf").exists()
    assert not published_plot_path.exists()

    show_payload = show_yiu_bundle(bundle_dir)

    assert show_payload.render_status == "not_requested"
    assert show_payload.available_renders == []
    assert show_payload.published_plot_artifact_path == str(published_plot_path.resolve())


def test_render_yiu_spec_force_overwrite_with_emit_renders_replaces_stale_published_plot(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    bundle_dir, _report = render_yiu_spec(spec_path, emit_renders=True)
    published_plot_path = workspace / "outputs" / "plot__demo_payload__payload_views.pdf"

    assert published_plot_path.exists()
    assert (bundle_dir / "payload_views.pdf").exists()

    overwritten_bundle_dir, _report = render_yiu_spec(spec_path, force_overwrite=True, emit_renders=True)
    inventory = _load_json(overwritten_bundle_dir / "visual_inventory.json")
    manifest = _load_json(overwritten_bundle_dir / "bundle_manifest.json")

    assert overwritten_bundle_dir == bundle_dir
    assert inventory["render_status"] == "rendered"
    assert inventory["render_count"] == 3
    assert manifest["render_status"] == "rendered"
    assert (bundle_dir / "payload_views.pdf").exists()
    assert published_plot_path.exists()


def test_render_yiu_spec_emit_renders_persists_failed_render_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    def _raise_render_failure(*args, **kwargs):
        raise RuntimeError("synthetic panel failure")

    monkeypatch.setattr(yiu_render_module, "render_view_panel", _raise_render_failure)

    with pytest.raises(YiuContractError, match="BaseRender failed for view 'payload'"):
        render_yiu_spec(spec_path, emit_renders=True)

    bundle_dir = workspace / "outputs" / "demo_payload"
    inventory = _load_json(bundle_dir / "visual_inventory.json")
    manifest = _load_json(bundle_dir / "bundle_manifest.json")

    assert inventory["render_status"] == "failed"
    assert inventory["render_count"] == 0
    assert inventory["last_rendered_at"] is None
    assert inventory["views"][0]["render_requested"] is True
    assert inventory["views"][0]["render_completed"] is False
    assert inventory["views"][1]["render_requested"] is False
    assert manifest["render_status"] == "failed"
    assert manifest["view_contracts"] == inventory["views"]


def test_render_yiu_spec_emit_renders_cleans_partial_artifacts_when_published_plot_copy_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    def _raise_copy_failure(src: Path, dst: Path, *args, **kwargs):
        Path(dst).write_bytes(b"%PDF-1.4\n% partial published copy\n")
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(yiu_render_module.shutil, "copyfile", _raise_copy_failure)

    with pytest.raises(YiuContractError, match="published-plot mirror failed"):
        render_yiu_spec(spec_path, emit_renders=True)

    bundle_dir = workspace / "outputs" / "demo_payload"
    inventory = _load_json(bundle_dir / "visual_inventory.json")
    manifest = _load_json(bundle_dir / "bundle_manifest.json")

    assert inventory["render_status"] == "failed"
    assert inventory["render_count"] == 3
    assert manifest["render_status"] == "failed"
    assert not (bundle_dir / "payload_views.pdf").exists()
    assert not (workspace / "outputs" / "plot__demo_payload__payload_views.pdf").exists()


def test_save_composite_render_adds_bottom_nucleotide_legend(tmp_path: Path) -> None:
    panel = np.full((48, 128, 4), 255, dtype=np.uint8)
    panel[:, :, 3] = 255
    panel[12:36, 20:108, :3] = np.array([191, 219, 254], dtype=np.uint8)
    render_path = tmp_path / "composite.png"

    save_composite_render(panel_images=[panel], render_path=render_path)
    composite = mpimg.imread(render_path)

    canonical = np.array([75, 85, 99], dtype=np.float32) / 255.0
    mismatch = np.array([185, 28, 28], dtype=np.float32) / 255.0
    legend_band = composite[int(composite.shape[0] * 0.75) :, :, :3]

    assert np.any(np.linalg.norm(legend_band - canonical, axis=2) < 0.05)
    assert np.any(np.linalg.norm(legend_band - mismatch, axis=2) < 0.05)


def test_composite_nucleotide_legend_centers_items_with_tight_square_swatch_spacing() -> None:
    fig, axis = plt.subplots(figsize=(4, 1), dpi=100)
    try:
        _draw_composite_nucleotide_legend(axis)
        legend = axis.get_legend()

        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["Canonical", "Mismatch"]
        assert all(text.get_text() != "A" for text in legend.get_texts())
        assert not axis.patches
        assert len(legend.legend_handles) == 2
        assert legend.legend_handles[0].get_marker() == "s"
        assert legend.legend_handles[1].get_marker() == "s"
        assert legend.legend_handles[0].get_color() == YIU_NUCLEOTIDE_LEGEND_CANONICAL_COLOR
        assert legend.legend_handles[1].get_color() == YIU_NUCLEOTIDE_LEGEND_MISMATCH_COLOR
        assert all(float(text.get_fontsize()) < 10.0 for text in legend.get_texts())
        assert math.isclose(
            float(legend.legend_handles[0].get_markersize()),
            float(legend.legend_handles[1].get_markersize()),
        )
        assert float(legend.columnspacing) <= 1.0

        fig.canvas.draw()
        bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(axis.transAxes.inverted())
        assert math.isclose(float(bbox.x0 + bbox.width / 2.0), 0.5, abs_tol=0.03)
        assert float(bbox.y0 + bbox.height / 2.0) < 0.45
    finally:
        plt.close(fig)
