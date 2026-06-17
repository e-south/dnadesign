"""Portfolio-level OPAL visual index assembly for TFBS Stage B reviews."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION
from .entries import visual_entries
from .io import read_json, read_realized_plot_manifest, require_existing_file
from .learning_loop import TfbsProbeQuestionLearningLoopSource, namespaced_learning_loop_visuals
from .specs import slug_token

REPLICATED_REVIEW_SCHEMA_VERSION = "stress_ethanol_cipro_growth.tfbs_stage_b_replicated_review.v1"
MIN_REPLICATE_COUNT = 2
EVIDENCE_TIER_LABELS = {
    "composition_campaign": "Composition campaigns",
    "placement_campaign": "Placement campaigns",
    "composition_learning_loop": "Learning-loop baselines",
    "placement_learning_loop": "Learning-loop baselines",
    "control_diagnostic": "Control diagnostics",
    "historical_precedent": "Historical precedent",
}
EVIDENCE_TIER_RANKS = {
    "composition_campaign": 10,
    "placement_campaign": 20,
    "composition_learning_loop": 40,
    "placement_learning_loop": 41,
    "control_diagnostic": 70,
    "historical_precedent": 90,
}
_TIER_PROFILE_ROLES = {
    "composition_campaign": {"canonical_stage_b_probe"},
    "placement_campaign": {
        "boundary_stage_b_count_fixed_minimal_placement_probe",
        "boundary_stage_b_count_fixed_sentinel_probe",
    },
}
_CLAIM_READY_REQUIRED_TIERS = frozenset({"composition_campaign"})


@dataclass(frozen=True)
class TfbsProbeQuestionReviewSource:
    """One replicated review for a named probe question in an OPAL portfolio notebook."""

    question_id: str
    question_label: str
    evidence_tier: str
    review_summary_json_path: Path | str


@dataclass(frozen=True)
class TfbsStageBReviewPortfolioResult:
    """Materialized OPAL collection artifacts for a review portfolio."""

    collection_manifest_path: Path
    collection_visual_index_path: Path
    comparison_set_count: int
    visual_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection_manifest_path": str(self.collection_manifest_path),
            "collection_visual_index_path": str(self.collection_visual_index_path),
            "comparison_set_count": int(self.comparison_set_count),
            "visual_count": int(self.visual_count),
        }


def write_tfbs_stage_b_review_portfolio(
    sources: Iterable[TfbsProbeQuestionReviewSource],
    *,
    out_dir: str | Path,
    collection_id: str,
    learning_loop_sources: Iterable[TfbsProbeQuestionLearningLoopSource] = (),
) -> TfbsStageBReviewPortfolioResult:
    """Write a combined OPAL collection manifest and visual index for replicated TFBS reviews."""

    source_rows = list(sources)
    if not source_rows:
        raise ValueError("TFBS Stage B review portfolio requires at least one probe question")
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_manifest_path = output_dir / "campaign_collection.json"
    collection_visual_index_path = output_dir / "collection_visual_manifest.json"

    visuals: list[dict[str, Any]] = []
    for source in source_rows:
        visuals.extend(_namespaced_source_visuals(source))
    for source in list(learning_loop_sources):
        visuals.extend(
            namespaced_learning_loop_visuals(
                source,
                evidence_tier_labels=EVIDENCE_TIER_LABELS,
                evidence_tier_ranks=EVIDENCE_TIER_RANKS,
            )
        )
    _fail_on_duplicate_visual_keys(visuals)
    comparison_sets = _comparison_sets_from_visuals(visuals)
    evidence_tiers = _evidence_tiers_from_visuals(visuals)

    collection_manifest = {
        "schema_version": "opal.campaign_collection.v2",
        "collection_id": collection_id,
        "dimensions": [{"id": "target", "label": "TFBS label"}],
        "relationships": [],
        "comparison_views": [],
        "collection_visual_surface_kinds": _surface_kinds(visuals),
        "evidence_tiers": evidence_tiers,
    }
    visual_index = {
        "schema_version": COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "collection_id": collection_id,
        "output_dir": str(output_dir),
        "surface_kinds": _surface_kinds(visuals),
        "evidence_tiers": evidence_tiers,
        "comparison_set_count": len(comparison_sets),
        "comparison_sets": comparison_sets,
        "visual_count": len(visuals),
        "visuals": visuals,
    }
    _write_json(collection_manifest_path, collection_manifest)
    _write_json(collection_visual_index_path, visual_index)
    return TfbsStageBReviewPortfolioResult(
        collection_manifest_path=collection_manifest_path,
        collection_visual_index_path=collection_visual_index_path,
        comparison_set_count=len(comparison_sets),
        visual_count=len(visuals),
    )


def _namespaced_source_visuals(source: TfbsProbeQuestionReviewSource) -> list[dict[str, Any]]:
    question_id = _required_token(source.question_id, field="question_id")
    question_label = _required_text(source.question_label, field="question_label")
    evidence_tier = _required_evidence_tier(source.evidence_tier)
    summary_path = Path(source.review_summary_json_path)
    summary = _replicated_review_summary(summary_path)
    _validate_evidence_tier_contract(summary, evidence_tier=evidence_tier, path=summary_path)
    plot_manifest_path = Path(str(summary["plot_manifest_json_path"]))
    trajectory_path = Path(str(summary["trajectory_csv_path"]))
    pair_summary_path = Path(str(summary["replicate_pair_summary_csv_path"]))
    require_existing_file(trajectory_path, role="replicated trajectory CSV")
    require_existing_file(pair_summary_path, role="replicated pair summary CSV")
    plot_manifest = read_realized_plot_manifest(plot_manifest_path)
    raw_visuals = visual_entries(
        plot_manifest=plot_manifest,
        plot_manifest_path=plot_manifest_path,
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
    )
    namespace = slug_token(question_id)
    out: list[dict[str, Any]] = []
    for raw in raw_visuals:
        comparison_set_key = f"{namespace}__{raw['comparison_set_key']}"
        visual = dict(raw)
        visual["visual_id"] = f"{namespace}__{raw.get('visual_id') or slug_token(str(raw.get('label') or 'visual'))}"
        visual["comparison_set_key"] = comparison_set_key
        visual["comparison_set_label"] = f"{question_label}: {raw['comparison_set_label']}"
        visual["comparison_set_match"] = {
            "probe_question_id": question_id,
            "evidence_tier": evidence_tier,
            **dict(raw.get("comparison_set_match") or {}),
        }
        visual["probe_question_id"] = question_id
        visual["probe_question_label"] = question_label
        visual["evidence_tier"] = evidence_tier
        visual["evidence_tier_label"] = EVIDENCE_TIER_LABELS[evidence_tier]
        visual["evidence_tier_rank"] = EVIDENCE_TIER_RANKS[evidence_tier]
        visual["source_review_summary_json_path"] = str(summary_path)
        visual["replicate_count"] = int(summary["replicate_count"])
        visual["replicate_seeds"] = list(summary["replicate_seeds"])
        _apply_evidence_tier_narrative(visual, question_label=question_label, evidence_tier=evidence_tier)
        out.append(visual)
    return out


def _replicated_review_summary(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    if summary.get("schema_version") != REPLICATED_REVIEW_SCHEMA_VERSION:
        raise ValueError(f"Unsupported TFBS replicated review schema: {summary.get('schema_version')!r}")
    if summary.get("status") != "PASS":
        raise ValueError(f"TFBS review portfolio only accepts PASS reviews: {path}")
    replicate_count = int(summary.get("replicate_count") or 0)
    if replicate_count < MIN_REPLICATE_COUNT:
        raise ValueError(
            f"TFBS review portfolio requires replicated review sources; {path} has replicate_count={replicate_count}"
        )
    seeds = summary.get("replicate_seeds")
    if not isinstance(seeds, list) or len(seeds) != replicate_count:
        raise ValueError(f"TFBS review portfolio source has inconsistent replicate_seeds: {path}")
    target_profile = summary.get("target_profile")
    if not isinstance(target_profile, Mapping):
        raise ValueError(f"TFBS review portfolio source is missing target_profile: {path}")
    interpretation_boundary = str(summary.get("interpretation_boundary") or "").strip()
    if not interpretation_boundary:
        interpretation_boundary = str(target_profile.get("interpretation_boundary") or "").strip()
        if interpretation_boundary:
            summary = dict(summary)
            summary["interpretation_boundary"] = interpretation_boundary
    if not interpretation_boundary:
        raise ValueError(f"TFBS review portfolio source is missing interpretation_boundary: {path}")
    for key in ("trajectory_csv_path", "replicate_pair_summary_csv_path", "plot_manifest_json_path"):
        value = str(summary.get(key) or "").strip()
        if not value:
            raise ValueError(f"TFBS review portfolio source is missing {key}: {path}")
    return summary


def _validate_evidence_tier_contract(summary: Mapping[str, Any], *, evidence_tier: str, path: Path) -> None:
    allowed_profile_roles = _TIER_PROFILE_ROLES.get(evidence_tier)
    if allowed_profile_roles is None:
        return
    target_profile = summary.get("target_profile")
    if not isinstance(target_profile, Mapping):
        raise ValueError(f"TFBS review portfolio current-tier source is missing target_profile: {path}")
    profile_role = str(target_profile.get("profile_role") or "").strip()
    if profile_role not in allowed_profile_roles:
        raise ValueError(
            "TFBS review portfolio evidence tier/profile_role mismatch: "
            f"tier={evidence_tier!r} profile_role={profile_role!r} source={path}"
        )
    if evidence_tier not in _CLAIM_READY_REQUIRED_TIERS:
        return
    claim_readiness = summary.get("claim_readiness")
    if not isinstance(claim_readiness, Mapping):
        raise ValueError(f"TFBS review portfolio current-tier source is missing claim_readiness: {path}")
    ready_claim_count = int(claim_readiness.get("ready_claim_count") or 0)
    blocked_or_limited_count = int(claim_readiness.get("blocked_or_limited_claim_count") or 0)
    if ready_claim_count < 1 or blocked_or_limited_count:
        raise ValueError(
            "TFBS review portfolio current-tier source is not claim-ready: "
            f"tier={evidence_tier!r} ready={ready_claim_count} "
            f"blocked_or_limited={blocked_or_limited_count} source={path}"
        )


def _comparison_sets_from_visuals(visuals: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for visual in visuals:
        key = str(visual.get("comparison_set_key") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "key": key,
                "label": str(visual.get("comparison_set_label") or key),
                "evidence_tier": str(visual.get("evidence_tier") or ""),
                "evidence_tier_label": str(visual.get("evidence_tier_label") or ""),
                "match": dict(visual.get("comparison_set_match") or {}),
            }
        )
    return rows


def _evidence_tiers_from_visuals(visuals: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tiers = {str(visual.get("evidence_tier") or "") for visual in visuals}
    return [
        {
            "id": tier,
            "label": EVIDENCE_TIER_LABELS[tier],
            "rank": EVIDENCE_TIER_RANKS[tier],
        }
        for tier in sorted(tiers, key=lambda item: EVIDENCE_TIER_RANKS[item])
        if tier in EVIDENCE_TIER_LABELS
    ]


def _surface_kinds(visuals: list[Mapping[str, Any]]) -> list[str]:
    return sorted({str(visual.get("surface_kind") or "").strip() for visual in visuals if visual.get("surface_kind")})


def _apply_evidence_tier_narrative(visual: dict[str, Any], *, question_label: str, evidence_tier: str) -> None:
    if evidence_tier != "control_diagnostic":
        return
    visual["premise"] = (
        f"Diagnostic check: {question_label} documents a control or confound check, not the main composition or "
        "placement result."
    )
    visual["claim_boundary"] = (
        "Diagnostic only: use this surface to explain probe limitations, boundary cases, and design choices; do not "
        "treat it as clean negative-control evidence."
    )
    visual["interpretation_note"] = (
        "This diagnostic remains visible to show known confounds and boundary cases; it is not part of the "
        "current claim tier."
    )


def _fail_on_duplicate_visual_keys(visuals: list[Mapping[str, Any]]) -> None:
    visual_ids = [str(visual.get("visual_id") or "") for visual in visuals]
    duplicate_visual_ids = sorted({item for item in visual_ids if item and visual_ids.count(item) > 1})
    if duplicate_visual_ids:
        raise ValueError(f"TFBS review portfolio has duplicate visual_id values: {duplicate_visual_ids}")


def _required_token(value: str, *, field: str) -> str:
    text = _required_text(value, field=field)
    token = slug_token(text)
    if token != text:
        raise ValueError(f"TFBS review portfolio {field} must already be a slug token: {value!r}")
    return token


def _required_evidence_tier(value: str) -> str:
    tier = str(value or "").strip()
    if tier not in EVIDENCE_TIER_LABELS:
        raise ValueError(
            f"TFBS review portfolio evidence_tier must be one of {sorted(EVIDENCE_TIER_LABELS)}; got {value!r}"
        )
    return tier


def _required_text(value: str, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"TFBS review portfolio {field} must be nonempty")
    return text


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "TfbsProbeQuestionLearningLoopSource",
    "TfbsStageBReviewPortfolioResult",
    "TfbsProbeQuestionReviewSource",
    "write_tfbs_stage_b_review_portfolio",
]
