"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/plan.py

Deliverable-plan loading for Retron hairpin review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...catalog.strict_mapping_io import DuplicateMappingKeyError, load_unique_yaml
from ...compiler.exceptions import RetronMsdCompilerError
from ..pwm.retention import PwmMotifOccurrence, validate_declared_trim_windows


@dataclass(frozen=True)
class PwmTrimPanel:
    payload_trim_id: str
    label: str
    retained_start_0: int
    retained_end_0: int
    trim_5p_nt: int
    trim_3p_nt: int
    retained_information_fraction: float


@dataclass(frozen=True)
class TetoReviewPlan:
    plan_path: Path
    design_set_path: Path
    compiler_spec_path: Path
    meme_pwm_path: Path
    parent_payload_sequence: str
    motif_occurrences: tuple[PwmMotifOccurrence, ...]
    deliverable_plan_id: str
    expected_variant_count: int
    pwm_panels: tuple[PwmTrimPanel, ...]


def load_teto_review_plan(path: Path, *, repo_root: Path) -> TetoReviewPlan:
    plan_path = path.expanduser().resolve()
    plan = _load_mapping(plan_path, label="Retron review deliverable plan")
    if plan.get("contract") != "retron_hairpin_deliverable_plan_v1":
        raise RetronMsdCompilerError(f"Unexpected Retron deliverable plan contract in {plan_path}")
    plan_id = str(plan.get("deliverable_plan_id") or "").strip()
    if plan_id != "teto_pwm_trim_rescue_v1":
        raise RetronMsdCompilerError(f"Unsupported Retron review deliverable plan id: {plan_id or '<missing>'}")

    design_set_path = _repo_path(repo_root, plan.get("design_set_ref"), field="design_set_ref")
    design_set = _load_mapping(design_set_path, label="Retron review design set")
    if design_set.get("contract") != "retron_msd_design_set_v1":
        raise RetronMsdCompilerError(f"Unexpected Retron design-set contract in {design_set_path}")
    expected_count = int(design_set.get("expected_variant_count") or 0)
    if expected_count <= 0:
        raise RetronMsdCompilerError(f"Retron design set has invalid expected_variant_count: {design_set_path}")

    families = plan.get("artifact_families")
    if not isinstance(families, Mapping):
        raise RetronMsdCompilerError(f"Retron deliverable plan is missing artifact_families: {plan_path}")
    pwm_family = families.get("pwm_trim_review_panel")
    if not isinstance(pwm_family, Mapping):
        raise RetronMsdCompilerError(f"Retron deliverable plan is missing pwm_trim_review_panel: {plan_path}")
    source_refs = _require_mapping(plan.get("source_refs"), "deliverable source_refs")
    meme_pwm_path = _repo_path(repo_root, source_refs.get("meme_pwm"), field="source_refs.meme_pwm")
    payload_trims = _require_mapping(design_set.get("payload_trims"), "design-set payload_trims")
    panels = tuple(
        _parse_pwm_panel(panel, payload_trims=payload_trims)
        for panel in _require_sequence(pwm_family.get("panels"), "PWM panels")
    )
    design_trim_ids = set(payload_trims)
    panel_trim_ids = {panel.payload_trim_id for panel in panels}
    if panel_trim_ids != design_trim_ids:
        raise RetronMsdCompilerError(
            "Retron PWM triptych payload_trim_id set does not match design-set payload_trims: "
            f"{sorted(panel_trim_ids)} != {sorted(design_trim_ids)}"
        )
    full_payload = _find_full_payload_trim(payload_trims)
    parent_payload = _require_mapping(design_set.get("parent_payload"), "design-set parent_payload")
    parent_sequence = str(parent_payload.get("source_sequence_5to3") or "").strip().upper()
    if parent_sequence != str(full_payload.get("exact_sequence_5to3") or "").strip().upper():
        raise RetronMsdCompilerError(
            "Retron review parent_payload.source_sequence_5to3 must match the untrimmed payload exact_sequence_5to3"
        )
    motif_occurrences = tuple(
        PwmMotifOccurrence.from_mapping(_require_mapping(raw, "parent_payload motif_occurrence"))
        for raw in _require_sequence(parent_payload.get("motif_occurrences"), "parent_payload motif_occurrences")
    )
    validate_declared_trim_windows(
        panels,
        parent_length=len(parent_sequence),
        motif_occurrences=motif_occurrences,
        meme_pwm_path=meme_pwm_path,
    )

    return TetoReviewPlan(
        plan_path=plan_path,
        design_set_path=design_set_path,
        compiler_spec_path=_repo_path(repo_root, plan.get("compiler_spec_ref"), field="compiler_spec_ref"),
        meme_pwm_path=meme_pwm_path,
        parent_payload_sequence=parent_sequence,
        motif_occurrences=motif_occurrences,
        deliverable_plan_id=plan_id,
        expected_variant_count=expected_count,
        pwm_panels=panels,
    )


def _parse_pwm_panel(raw: object, *, payload_trims: Mapping[str, Any]) -> PwmTrimPanel:
    panel = _require_mapping(raw, "PWM panel")
    span = _require_mapping(panel.get("retained_parent_span_0"), "PWM panel retained_parent_span_0")
    payload_trim_id = str(panel.get("payload_trim_id") or "").strip()
    trim = _require_mapping(payload_trims.get(payload_trim_id), f"design-set {payload_trim_id} payload trim")
    return PwmTrimPanel(
        payload_trim_id=payload_trim_id,
        label=str(panel.get("label") or "").strip(),
        retained_start_0=int(span.get("start")),
        retained_end_0=int(span.get("end")),
        trim_5p_nt=int(trim.get("trim_5p_nt")),
        trim_3p_nt=int(trim.get("trim_3p_nt")),
        retained_information_fraction=float(trim.get("retained_information_fraction")),
    )


def _find_full_payload_trim(payload_trims: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = []
    for trim_id, raw_trim in payload_trims.items():
        trim = _require_mapping(raw_trim, f"design-set {trim_id} payload trim")
        span = _require_mapping(trim.get("retained_parent_span_0"), f"design-set {trim_id} retained_parent_span_0")
        start = int(span.get("start"))
        end = int(span.get("end"))
        sequence = str(trim.get("exact_sequence_5to3") or "")
        if start == 0 and end == len(sequence):
            candidates.append(trim)
    if len(candidates) != 1:
        raise RetronMsdCompilerError(
            f"Retron design set must declare exactly one untrimmed payload, found {len(candidates)}"
        )
    return candidates[0]


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RetronMsdCompilerError(f"{label} not found: {path}")
    try:
        payload = load_unique_yaml(path)
    except DuplicateMappingKeyError as exc:
        raise RetronMsdCompilerError(f"{label} contains {exc}: {path}") from exc
    if not isinstance(payload, dict):
        raise RetronMsdCompilerError(f"{label} must be a mapping: {path}")
    return payload


def _repo_path(repo_root: Path, raw: object, *, field: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise RetronMsdCompilerError(f"Retron deliverable plan is missing {field}")
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _require_mapping(raw: object, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise RetronMsdCompilerError(f"Retron review output expected mapping for {label}")
    return raw


def _require_sequence(raw: object, label: str) -> list[object]:
    if not isinstance(raw, list):
        raise RetronMsdCompilerError(f"Retron review output expected list for {label}")
    return raw


__all__ = ["PwmTrimPanel", "TetoReviewPlan", "load_teto_review_plan"]
