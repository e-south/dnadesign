"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/round_context.py

RoundContext: lightweight per-round context passed to objectives and used
for artifacts/provenance. Not persisted per-row; written once per round.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


def _git_short_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return "nogit"


def compute_fingerprint(payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Compute a stable SHA-256 over a JSON-serialized payload (sorted keys).
    Returns dict with 'full' and 'short' (first 10 chars).
    """
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    h = hashlib.sha256(raw).hexdigest()
    return {"full": h, "short": h[:10]}


@dataclass
class RoundContext:
    slug: str
    round_index: int
    run_id: str
    setpoint: List[float]  # length 4
    label_ids: List[str]
    training_label_count: int
    effect_pool_for_scaling: List[float]
    percentile_cfg: Dict[str, Any]
    model_name: str
    model_params: Dict[str, Any]
    x_transform: Dict[str, Any]
    y_ingest_transform: Dict[str, Any]
    y_expected_length: Optional[int]
    code_version: str
    fingerprint_full: str
    fingerprint_short: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def build(
        *,
        slug: str,
        round_index: int,
        setpoint: List[float],
        label_ids: List[str],
        effect_pool_for_scaling: List[float],
        percentile_cfg: Dict[str, Any],
        model_name: str,
        model_params: Dict[str, Any],
        x_transform: Dict[str, Any],
        y_ingest_transform: Dict[str, Any],
        y_expected_length: Optional[int],
        run_id: str,
    ) -> "RoundContext":
        code_version = _git_short_sha()
        payload = {
            "slug": slug,
            "round": round_index,
            "setpoint": list(map(float, setpoint)),
            "label_ids": [str(i) for i in label_ids],
            "effect_pool_len": len(effect_pool_for_scaling),
            "percentile_cfg": percentile_cfg,
            "model": {"name": model_name, "params": model_params},
            "x_transform": x_transform,
            "y_ingest_transform": y_ingest_transform,
            "y_expected_length": y_expected_length,
            "code_version": code_version,
        }
        fp = compute_fingerprint(payload)
        return RoundContext(
            slug=slug,
            round_index=round_index,
            run_id=run_id,
            setpoint=list(map(float, setpoint)),
            label_ids=[str(i) for i in label_ids],
            training_label_count=len(label_ids),
            effect_pool_for_scaling=list(map(float, effect_pool_for_scaling)),
            percentile_cfg=percentile_cfg,
            model_name=model_name,
            model_params=model_params,
            x_transform=x_transform,
            y_ingest_transform=y_ingest_transform,
            y_expected_length=y_expected_length,
            code_version=code_version,
            fingerprint_full=fp["full"],
            fingerprint_short=fp["short"],
        )
