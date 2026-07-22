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
from .benchling_import import BenchlingGenbankImportPlan, parse_benchling_genbank_import_plan
from .pwm_trim import PwmTrimPanel, parse_pwm_trim_context
from .review_variant_ids import parse_review_variant_ids


@dataclass(frozen=True)
class RetronReviewPlan:
    plan_path: Path
    design_set_path: Path
    compiler_spec_path: Path
    meme_pwm_path: Path
    preferred_generated_root: Path
    preferred_materialized_root: Path
    parent_payload_sequence: str
    motif_occurrences: tuple[object, ...]
    deliverable_plan_id: str
    expected_variant_count: int
    pwm_panels: tuple[PwmTrimPanel, ...]
    review_variant_ids: Mapping[str, str]
    benchling_import: BenchlingGenbankImportPlan


def load_retron_review_plan(path: Path, *, repo_root: Path) -> RetronReviewPlan:
    plan_path = path.expanduser().resolve()
    plan = _load_mapping(plan_path, label="Retron review deliverable plan")
    if plan.get("contract") != "retron_hairpin_deliverable_plan_v1":
        raise RetronMsdCompilerError(f"Unexpected Retron deliverable plan contract in {plan_path}")
    plan_id = str(plan.get("deliverable_plan_id") or "").strip()
    if not plan_id:
        raise RetronMsdCompilerError(f"Retron deliverable plan is missing deliverable_plan_id: {plan_path}")
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
    benchling_import = parse_benchling_genbank_import_plan(families)
    review_variant_ids = parse_review_variant_ids(families, design_set=design_set, benchling_import=benchling_import)
    pwm_family = families.get("pwm_trim_review_panel")
    if not isinstance(pwm_family, Mapping):
        raise RetronMsdCompilerError(f"Retron deliverable plan is missing pwm_trim_review_panel: {plan_path}")
    source_refs = _require_mapping(plan.get("source_refs"), "deliverable source_refs")
    meme_pwm_path = _repo_path(repo_root, source_refs.get("meme_pwm"), field="source_refs.meme_pwm")
    output_policy = _require_mapping(plan.get("output_policy"), "deliverable output_policy")
    preferred_generated_root = _repo_path(
        repo_root,
        output_policy.get("preferred_generated_root"),
        field="output_policy.preferred_generated_root",
    )
    preferred_materialized_root = _repo_path(
        repo_root,
        output_policy.get("preferred_materialized_root"),
        field="output_policy.preferred_materialized_root",
    )
    pwm_context = parse_pwm_trim_context(
        pwm_family=pwm_family,
        design_set=design_set,
        meme_pwm_path=meme_pwm_path,
    )
    return RetronReviewPlan(
        plan_path=plan_path,
        design_set_path=design_set_path,
        compiler_spec_path=_repo_path(repo_root, plan.get("compiler_spec_ref"), field="compiler_spec_ref"),
        meme_pwm_path=meme_pwm_path,
        preferred_generated_root=preferred_generated_root,
        preferred_materialized_root=preferred_materialized_root,
        parent_payload_sequence=pwm_context.parent_payload_sequence,
        motif_occurrences=pwm_context.motif_occurrences,
        deliverable_plan_id=plan_id,
        expected_variant_count=expected_count,
        pwm_panels=pwm_context.panels,
        review_variant_ids=review_variant_ids,
        benchling_import=benchling_import,
    )


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


__all__ = [
    "BenchlingGenbankImportPlan",
    "PwmTrimPanel",
    "RetronReviewPlan",
    "load_retron_review_plan",
]
