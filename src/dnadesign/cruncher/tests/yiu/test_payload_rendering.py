"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/yiu/test_payload_rendering.py

Runtime contracts for the payload-centric YIU v4 lane.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

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
from dnadesign.cruncher.yiu.candidate_generation import CandidatePlan, MutationChoice
from dnadesign.cruncher.yiu.errors import NoFeasiblePlanError, YiuContractError
from dnadesign.cruncher.yiu.load import load_yiu_spec
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
from dnadesign.cruncher.yiu.publish_layout import build_published_artifacts, resolve_payload_bundle_layout
from dnadesign.cruncher.yiu.scoring import CandidateScore
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
    "midpoint_proximity",
    "body_length_balance",
    "terminal_position_avoidance",
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
    max_payload_body_length: int = 12,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": mode,
        "overhang_length": 4,
        "max_payload_body_length": max_payload_body_length,
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
    pwm_mode: str = "none",
    pwm_source: dict[str, object] | None = None,
    emit_render_jobs_debug: bool = False,
) -> dict[str, object]:
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
            "mismatches": {
                "count": mismatch_count,
                "candidate_positions": candidate_positions or [1, 2],
                "allowed_strands": ["complement", "payload"],
                "strand_mode": "per_position",
                "default_strand_preference": "complement",
            },
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
    source_artifact: str | None = None,
    metadata: dict[str, object] | None = None,
    junction_mode: str = "derived",
    junction_start: int = TOY_JUNCTION_START,
    junction_end: int = TOY_JUNCTION_END,
    mismatch_count: int = 1,
    candidate_positions: list[int] | None = None,
    pwm_mode: str = "none",
    pwm_source: dict[str, object] | None = None,
) -> dict[str, object]:
    sample_hit: dict[str, object] = {"hit_id": hit_id, "sample_name": sample_name}
    if payload_sequence is not None:
        sample_hit["payload_sequence"] = payload_sequence
    if source_artifact_path is not None:
        sample_hit["source_artifact_path"] = source_artifact_path
    if source_artifact is not None:
        sample_hit["source_artifact"] = source_artifact
    if metadata is not None:
        sample_hit["metadata"] = metadata
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
            "mismatches": {
                "count": mismatch_count,
                "candidate_positions": candidate_positions or [1, 2],
                "allowed_strands": ["complement", "payload"],
                "strand_mode": "per_position",
                "default_strand_preference": "complement",
            },
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
    body_length_balance: int = 0,
    terminal_positions_used: int = 0,
) -> CandidatePlan:
    return CandidatePlan(
        junction_start=4,
        junction_end=8,
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
        midpoint_distance=midpoint_distance,
        body_length_balance=body_length_balance,
        terminal_positions_used=terminal_positions_used,
        default_strand_preference_count=default_strand_preference_count,
        lexical_key=lexical_key,
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

    assert normalized.contract == "yiu_normalized_payload_v4"
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


def test_normalize_derived_mode_selects_midpoint_nearest_internal_window(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "derived_payload.yiu.yaml"
    _write_yaml(
        spec_path, _user_sequence_spec(sequence="AACCGGTTGGTT", junction_mode="derived", candidate_positions=[1])
    )

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    assert normalized.junction.mode == "derived"
    assert normalized.junction.start == 4
    assert normalized.junction.end == 8


def test_normalize_optimize_mode_raises_no_feasible_plan_for_tight_body_bound(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspace" / "configs" / "yiu" / "no_plan.yiu.yaml"
    payload = _user_sequence_spec(sequence="AACCGGTTA", junction_mode="optimize")
    payload["optimization"]["junction"]["max_payload_body_length"] = 1
    _write_yaml(spec_path, payload)

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    with pytest.raises(NoFeasiblePlanError, match="No feasible optimized junction found"):
        normalize_payload(spec, workspace_root=workspace_root)


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
            metadata={
                "source_workspace": "demo_monotypic_tetr",
                "source_artifact": "outputs/optimize/tables/elites.parquet",
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
            metadata={
                "source_workspace": "missing_workspace",
                "source_artifact": "outputs/optimize/tables/elites.parquet",
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
    result = select_best_candidate(
        candidates=(candidate_a, candidate_b),
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
        scorable_motifs=(),
        pwm_effective=True,
    )

    assert result.winner.lexical_key == "b"
    assert result.score.total_loss == 1.0


def test_select_best_candidate_breaks_pwm_tie_on_midpoint_proximity(monkeypatch: pytest.MonkeyPatch) -> None:
    farther = _candidate(
        lexical_key="farther",
        default_strand_preference_count=1,
        midpoint_distance=2,
        body_length_balance=0,
        terminal_positions_used=0,
    )
    closer = _candidate(
        lexical_key="closer",
        default_strand_preference_count=1,
        midpoint_distance=1,
        body_length_balance=9,
        terminal_positions_used=1,
    )

    monkeypatch.setattr(
        yiu_optimizer_module,
        "score_candidate",
        lambda **_: CandidateScore(worst_loss=1.0, total_loss=1.0),
    )
    result = select_best_candidate(
        candidates=(farther, closer),
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
        scorable_motifs=(),
        pwm_effective=True,
    )

    assert result.winner.lexical_key == "closer"


def test_select_best_candidate_breaks_pwm_tie_on_body_length_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    less_balanced = _candidate(
        lexical_key="less-balanced",
        default_strand_preference_count=1,
        midpoint_distance=1,
        body_length_balance=3,
        terminal_positions_used=0,
    )
    more_balanced = _candidate(
        lexical_key="more-balanced",
        default_strand_preference_count=1,
        midpoint_distance=1,
        body_length_balance=1,
        terminal_positions_used=1,
    )

    monkeypatch.setattr(
        yiu_optimizer_module,
        "score_candidate",
        lambda **_: CandidateScore(worst_loss=1.0, total_loss=1.0),
    )
    result = select_best_candidate(
        candidates=(less_balanced, more_balanced),
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
        scorable_motifs=(),
        pwm_effective=True,
    )

    assert result.winner.lexical_key == "more-balanced"


def test_select_best_candidate_breaks_pwm_tie_on_terminal_position_avoidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal_heavy = _candidate(
        lexical_key="terminal-heavy",
        default_strand_preference_count=1,
        midpoint_distance=1,
        body_length_balance=1,
        terminal_positions_used=1,
    )
    terminal_safe = _candidate(
        lexical_key="terminal-safe",
        default_strand_preference_count=0,
        midpoint_distance=1,
        body_length_balance=1,
        terminal_positions_used=0,
    )

    monkeypatch.setattr(
        yiu_optimizer_module,
        "score_candidate",
        lambda **_: CandidateScore(worst_loss=1.0, total_loss=1.0),
    )
    result = select_best_candidate(
        candidates=(terminal_heavy, terminal_safe),
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
        scorable_motifs=(),
        pwm_effective=True,
    )

    assert result.winner.lexical_key == "terminal-safe"


def test_select_best_candidate_without_pwm_prefers_default_strand_then_lexical_order() -> None:
    preferred = _candidate(lexical_key="z", default_strand_preference_count=1)
    nonpreferred = _candidate(lexical_key="a", default_strand_preference_count=0)

    result = select_best_candidate(
        candidates=(nonpreferred, preferred),
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
        scorable_motifs=(),
        pwm_effective=False,
    )

    assert result.winner.lexical_key == "z"

    lexical_a = _candidate(lexical_key="a", default_strand_preference_count=1)
    lexical_b = _candidate(lexical_key="b", default_strand_preference_count=1)
    lexical_result = select_best_candidate(
        candidates=(lexical_b, lexical_a),
        reference_payload_sequence=TOY_SEQUENCE,
        reference_complement_sequence="".join(reverse_complement_iupac(base) for base in TOY_SEQUENCE),
        scorable_motifs=(),
        pwm_effective=False,
    )
    assert lexical_result.winner.lexical_key == "a"


def test_render_yiu_spec_publishes_v4_bundle_and_payload_visual_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    bundle_dir, report = render_yiu_spec(spec_path)

    assert report.contract == "split_yiu_payload_rendering_v4"
    manifest = _load_json(bundle_dir / "bundle_manifest.json")
    normalized = _load_json(bundle_dir / "normalized_payload.json")
    inventory = _load_json(bundle_dir / "visual_inventory.json")
    payload_view = _load_json(bundle_dir / "payload_view.json")
    split_rows = _load_jsonl(bundle_dir / "split_payload_view.json")
    assembled_view = _load_json(bundle_dir / "assembled_payload_view.json")

    assert manifest["bundle_contract"] == "split_yiu_payload_bundle_v4"
    assert manifest["published_plot_artifact_path"] == "outputs/plot__demo_payload__payload_views.pdf"
    assert normalized["contract"] == "yiu_normalized_payload_v4"
    assert inventory["pwm_effective"] is False
    assert inventory["published_plot_artifact_path"] == "outputs/plot__demo_payload__payload_views.pdf"
    assert payload_view["contract_kind"] == "yiu_payload_visual_v1"
    assert payload_view["motif_layers"] == []
    assert len(split_rows) == 2
    assert assembled_view["contract_kind"] == "sequence_evidence_map_v1"


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
    assert outcome.report.spec_name == "demo_payload"


def test_render_yiu_spec_with_pwm_effective_adds_payload_motif_layers_and_show_roundtrips(tmp_path: Path) -> None:
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


def test_split_and_assembled_views_center_titles() -> None:
    payload_overrides = build_yiu_style_overrides("payload")
    split_overrides = build_yiu_style_overrides("split_payload")
    assembled_overrides = build_yiu_style_overrides("assembled_payload")

    assert payload_overrides["figure_scale"] == split_overrides["figure_scale"] == assembled_overrides["figure_scale"]
    assert split_overrides["overlay_align"] == "center"
    assert assembled_overrides["overlay_align"] == "center"


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
    assert right_fragment.ghost_excised_context is not None

    meta = build_split_payload_row_meta(right_fragment)
    sticky_end_span = right_fragment.sticky_end_display_span.model_dump(mode="json")

    assert meta["fragment_side"] == "right"
    assert meta["selected_sticky_end_sequence_5to3"] == right_fragment.selected_sticky_end_sequence_5to3
    assert meta["canonical_sticky_end_sequence_5to3"] == right_fragment.canonical_sticky_end_sequence_5to3
    assert meta["sticky_end_display_span"] == sticky_end_span
    assert meta["payload_junction_window"] == right_fragment.payload_junction_window.model_dump(mode="json")
    assert meta["connector_hidden_indices"] == list(range(sticky_end_span["start"], sticky_end_span["end"]))
    assert meta["connector_cross_indices"] == []
    assert meta["connector_overhang_spans"] == [sticky_end_span]
    assert meta["ghost_excised_context"] == right_fragment.ghost_excised_context.model_dump(mode="json")
    assert meta["dim_base_indices"] == {
        "primary": list(right_fragment.ghost_excised_context.primary_indices),
        "complement": list(right_fragment.ghost_excised_context.complement_indices),
    }


def test_assembled_payload_view_metadata_preserves_junction_connector_contract(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec(mismatch_count=2, candidate_positions=[1, 2]))

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)

    meta = build_assembled_payload_view_meta(normalized)
    junction_span = {
        "start": normalized.junction.start,
        "end": normalized.junction.end,
        "coordinate_space": "payload_forward",
    }
    mismatch_indices = [site.payload_index for site in normalized.mismatches]

    assert meta["junction_span"] == junction_span
    assert meta["mismatches"] == [site.model_dump(mode="json") for site in normalized.mismatches]
    assert meta["base_highlights"] == {"primary": mismatch_indices, "complement": mismatch_indices}
    assert meta["connector_hidden_indices"] == [
        index
        for index in range(normalized.junction.start, normalized.junction.end)
        if index not in set(mismatch_indices)
    ]
    assert meta["connector_cross_indices"] == mismatch_indices
    assert meta["connector_overhang_spans"] == [junction_span]


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
    }
    assert payload_view["motif_layers"] == [layer.model_dump(mode="json") for layer in motif_layers]
    assert payload_view["mismatches"] == [entry.model_dump(mode="json") for entry in mismatch_annotations]
    assert payload_view["meta"] == meta


def test_publish_layout_tracks_relative_bundle_artifacts_and_view_entries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    spec_path = workspace / "configs" / "yiu" / "demo_payload.yiu.yaml"
    _write_yaml(spec_path, _user_sequence_spec())

    spec, _resolved_spec_path, workspace_root = load_yiu_spec(spec_path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)
    layout = resolve_payload_bundle_layout(workspace / spec.output.bundle_dir)
    published_artifacts = build_published_artifacts(
        layout=layout,
        published_plot_artifact_path=str(spec.output.published_plot_path),
    )
    view_entries = build_payload_view_entries(layout=layout, normalized=normalized)

    assert published_artifacts == {
        "normalized_payload": "normalized_payload.json",
        "bundle_manifest": "bundle_manifest.json",
        "visual_inventory": "visual_inventory.json",
        "payload_view": "payload_view.json",
        "split_payload_view": "split_payload_view.json",
        "assembled_payload_view": "assembled_payload_view.json",
        "payload_views_pdf": "payload_views.pdf",
        "published_plot_pdf": "outputs/plot__demo_payload__payload_views.pdf",
    }
    assert [entry.view_id for entry in view_entries] == ["payload", "split_payload", "assembled_payload"]
    assert [entry.view_contract_path for entry in view_entries] == [
        "payload_view.json",
        "split_payload_view.json",
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
    normalized_payload_dump = build_normalized_payload_dump(spec=spec, normalized=normalized, layout=layout)
    view_entries = build_payload_view_entries(layout=layout, normalized=normalized)
    inventory = build_payload_visual_inventory(
        spec=spec,
        normalized=normalized,
        layout=layout,
        view_entries=view_entries,
    )
    manifest = build_payload_bundle_manifest(normalized=normalized, inventory=inventory)

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
    normalized_dump = build_normalized_payload_dump(spec=spec, normalized=normalized, layout=layout)

    assert inventory.render_status == "not_requested"
    assert inventory.payload_view_requires_motif_layers is True
    assert manifest.render_status == inventory.render_status
    assert manifest.view_contracts == inventory.views
    assert manifest.composite_render_artifact_path == inventory.composite_render_artifact_path == "payload_views.pdf"
    assert manifest.published_plot_artifact_path == inventory.published_plot_artifact_path
    assert manifest.published_plot_artifact_path == "outputs/plot__pwm_payload__payload_views.pdf"
    assert normalized_dump["published_artifacts"]["payload_view"] == "payload_view.json"
    assert (
        normalized_dump["published_artifacts"]["published_plot_pdf"] == "outputs/plot__pwm_payload__payload_views.pdf"
    )


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
                view_contract_path="split_payload_view.json",
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
    assert payload_view["display"]["title"] == "TetR payload"
    assert payload_view["motif_layers"][0]["tf_name"] == "tetR"
    assert payload_view["motif_layers"][0]["reference_strand"] == "+"
    assert payload_view["motif_layers"][0]["start"] == 0
    assert payload_view["motif_layers"][0]["end"] == 17
    assert payload_view["meta"]["row_labels"] == {}
    for got, expected in zip(payload_view["motif_layers"][0]["matrix"], _canonical_tetr_pwm_rows(), strict=True):
        assert got == pytest.approx(expected)
    assert show_payload.pwm_effective is True
    assert show_payload.motif_context.effective is True
    assert show_payload.integrity.status == "ok"
    assert show_payload.published_plot_artifact_path == str(
        (workspace / "outputs" / "plots" / "plot__yiu__tetr_monotypic_hit__payload_views.pdf").resolve()
    )

    split_rows = _load_jsonl(bundle_dir / "split_payload_view.json")
    assert split_rows[0]["meta"]["row_labels"] == {}
    assert split_rows[1]["meta"]["row_labels"] == {}
    assert split_rows[0]["meta"]["dim_base_indices"] == {
        "primary": list(range(0, 7)),
        "complement": list(range(0, 11)),
    }
    assert split_rows[1]["meta"]["dim_base_indices"] == {
        "primary": list(range(7, 18)),
        "complement": list(range(11, 18)),
    }
    assembled_view = _load_json(bundle_dir / "assembled_payload_view.json")
    assert assembled_view["meta"]["row_labels"] == {}


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
    assert payload_view["display"]["title"] == "BaeR payload (3 sites)"
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
