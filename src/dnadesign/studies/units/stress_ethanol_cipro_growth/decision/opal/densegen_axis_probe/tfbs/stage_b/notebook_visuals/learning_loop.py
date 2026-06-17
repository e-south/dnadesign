"""Learning-loop baseline adapters for TFBS portfolio notebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..learning_loop_baselines.notebook_visuals import learning_loop_visual_entries
from .specs import slug_token


@dataclass(frozen=True)
class TfbsProbeQuestionLearningLoopSource:
    """One learning-loop baseline for a named probe question in an OPAL portfolio notebook."""

    question_id: str
    question_label: str
    evidence_tier: str
    replay_manifest_json_path: Path | str


def namespaced_learning_loop_visuals(
    source: TfbsProbeQuestionLearningLoopSource,
    *,
    evidence_tier_labels: Mapping[str, str],
    evidence_tier_ranks: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Return namespaced portfolio entries for one learning-loop baseline surface."""

    question_id = _required_token(source.question_id, field="question_id")
    question_label = _required_text(source.question_label, field="question_label")
    evidence_tier = _required_evidence_tier(source.evidence_tier, labels=evidence_tier_labels)
    manifest_path = Path(source.replay_manifest_json_path)
    manifest_tier = _manifest_visual_tier(manifest_path)
    if evidence_tier != manifest_tier:
        raise ValueError(
            "TFBS learning-loop evidence_tier must match the source manifest visual_tier: "
            f"requested={evidence_tier!r} manifest={manifest_tier!r}"
        )
    namespace = slug_token(question_id)
    out: list[dict[str, Any]] = []
    for raw in learning_loop_visual_entries(manifest_path):
        comparison_set_key = f"{namespace}__{raw['comparison_set_key']}"
        visual = dict(raw)
        visual["visual_id"] = f"{namespace}__{raw.get('visual_id') or slug_token(str(raw.get('label') or 'visual'))}"
        visual["comparison_set_key"] = comparison_set_key
        visual["comparison_set_label"] = _prefixed_label(question_label, str(raw["comparison_set_label"]))
        visual["comparison_set_match"] = {
            "probe_question_id": question_id,
            "evidence_tier": evidence_tier,
            **dict(raw.get("comparison_set_match") or {}),
        }
        visual["probe_question_id"] = question_id
        visual["probe_question_label"] = question_label
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


def _prefixed_label(source_label: str, raw_label: str) -> str:
    """Prefix namespaced learning-loop labels without duplicating identical text."""

    source = source_label.strip()
    raw = raw_label.strip()
    if raw == source or raw in {"Learning-loop baseline", "Baseline review"}:
        return source
    return f"{source}: {raw}"


__all__ = ["TfbsProbeQuestionLearningLoopSource", "namespaced_learning_loop_visuals"]
