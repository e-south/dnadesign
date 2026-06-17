"""Learning-loop baseline adapters for TFBS portfolio notebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..learning_loop_baselines.notebook_visuals import learning_loop_visual_entries
from .specs import slug_token


@dataclass(frozen=True)
class TfbsStageBLearningLoopPortfolioSource:
    """One learning-loop baseline review surface to expose in an OPAL portfolio notebook."""

    surface_id: str
    surface_label: str
    evidence_tier: str
    replay_manifest_json_path: Path | str


def namespaced_learning_loop_visuals(
    source: TfbsStageBLearningLoopPortfolioSource,
    *,
    evidence_tier_labels: Mapping[str, str],
    evidence_tier_ranks: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Return namespaced portfolio entries for one learning-loop baseline surface."""

    source_id = _required_token(source.surface_id, field="surface_id")
    source_label = _required_text(source.surface_label, field="surface_label")
    evidence_tier = _required_evidence_tier(source.evidence_tier, labels=evidence_tier_labels)
    manifest_path = Path(source.replay_manifest_json_path)
    manifest_tier = _manifest_visual_tier(manifest_path)
    if evidence_tier != manifest_tier:
        raise ValueError(
            "TFBS learning-loop evidence_tier must match the source manifest visual_tier: "
            f"requested={evidence_tier!r} manifest={manifest_tier!r}"
        )
    namespace = slug_token(source_id)
    out: list[dict[str, Any]] = []
    for raw in learning_loop_visual_entries(manifest_path):
        comparison_set_key = f"{namespace}__{raw['comparison_set_key']}"
        visual = dict(raw)
        visual["visual_id"] = f"{namespace}__{raw.get('visual_id') or slug_token(str(raw.get('label') or 'visual'))}"
        visual["comparison_set_key"] = comparison_set_key
        visual["comparison_set_label"] = f"{source_label}: {raw['comparison_set_label']}"
        visual["comparison_set_match"] = {
            "source_review_surface_id": source_id,
            "evidence_tier": evidence_tier,
            **dict(raw.get("comparison_set_match") or {}),
        }
        visual["source_review_surface_id"] = source_id
        visual["source_review_surface_label"] = source_label
        visual["source_review_summary_json_path"] = str(manifest_path)
        visual["evidence_tier"] = evidence_tier
        visual["evidence_tier_label"] = evidence_tier_labels[evidence_tier]
        visual["evidence_tier_rank"] = evidence_tier_ranks[evidence_tier]
        out.append(visual)
    return out


def _required_token(value: str, *, field: str) -> str:
    text = _required_text(value, field=field)
    token = slug_token(text)
    if token != text:
        raise ValueError(f"TFBS learning-loop portfolio {field} must already be a slug token: {value!r}")
    return token


def _required_text(value: str, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"TFBS learning-loop portfolio {field} must be nonempty")
    return text


def _required_evidence_tier(value: str, *, labels: Mapping[str, str]) -> str:
    tier = str(value or "").strip()
    if tier not in labels:
        raise ValueError(f"TFBS learning-loop evidence_tier must be one of {sorted(labels)}; got {value!r}")
    return tier


def _manifest_visual_tier(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tier = str(payload.get("visual_tier") or "").strip()
    if not tier:
        raise ValueError(f"TFBS learning-loop manifest is missing visual_tier: {path}")
    return tier


__all__ = ["TfbsStageBLearningLoopPortfolioSource", "namespaced_learning_loop_visuals"]
