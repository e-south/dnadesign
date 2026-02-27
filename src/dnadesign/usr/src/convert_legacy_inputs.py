"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/convert_legacy_inputs.py

Profile and input helper functions for legacy densegen conversion flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .errors import SchemaError, ValidationError


@dataclass(frozen=True)
class Profile:
    name: str
    expected_length: Optional[int]
    logits_key: str
    logits_dst: str
    logits_expected_dim: int
    densegen_plan: str


def profile_60bp_dual_promoter() -> Profile:
    return Profile(
        name="60bp_dual_promoter_cpxR_LexA",
        expected_length=60,
        logits_key="evo2_logits_mean_pooled",
        logits_dst="infer__evo2_7b__60bp_dual_promoter_cpxR_LexA__logits_mean",
        logits_expected_dim=512,
        densegen_plan="sigma70_mid",
    )


def _require_torch():
    try:
        import torch
    except ImportError as e:
        raise SchemaError("torch is required for legacy conversion (convert-legacy).") from e
    return torch


def _coerce_logits(v: Any, *, want_dim: int) -> Optional[List[float]]:
    """Return a flat list[float] of length want_dim, or None if unavailable."""
    if v is None:
        return None
    mod = getattr(v.__class__, "__module__", "")
    if mod.startswith("torch"):
        torch = _require_torch()
        if not isinstance(v, torch.Tensor):
            raise ValidationError(f"logits type mismatch (module={mod})")
        t = v.detach().cpu()
        if t.ndim == 2 and t.shape[0] == 1:
            t = t.reshape(t.shape[1])
        if t.ndim != 1 or int(t.shape[0]) != want_dim:
            raise ValidationError(f"logits shape mismatch (got {tuple(t.shape)}, expected [{want_dim}])")
        return t.to(dtype=torch.float32).tolist()
    if isinstance(v, (list, tuple)):
        arr = list(v)
        if len(arr) == 1 and isinstance(arr[0], (list, tuple)):
            arr = list(arr[0])
        if len(arr) != want_dim:
            raise ValidationError(f"logits length mismatch (got {len(arr)}, expected {want_dim})")
        return [float(x) for x in arr]
    return None


def _tf_from_parts(parts: Sequence[str]) -> List[str]:
    """Extract unique TF names from strings like 'lexa:...' or 'cpxr:...'"""
    out = set()
    for s in parts:
        if not isinstance(s, str):
            continue
        if ":" in s:
            tf = s.split(":", 1)[0].strip().lower()
            if tf:
                out.add(tf)
    return sorted(out)


def _count_tf(parts: Sequence[str]) -> Dict[str, int]:
    counts = {"cpxr": 0, "lexa": 0}
    for s in parts:
        if not isinstance(s, str):
            continue
        if s.lower().startswith("lexa:"):
            counts["lexa"] += 1
        elif s.lower().startswith("cpxr:"):
            counts["cpxr"] += 1
    return counts


def _ensure_pt_list_of_dicts(p: Path) -> List[dict]:
    torch = _require_torch()
    checkpoint = torch.load(str(p), map_location=torch.device("cpu"))
    if not isinstance(checkpoint, list) or not checkpoint:
        raise SchemaError(f"{p} must be a non-empty list.")
    for i, entry in enumerate(checkpoint):
        if not isinstance(entry, dict):
            raise SchemaError(f"{p} entry {i} is not a dict.")
        if "sequence" not in entry:
            raise SchemaError(f"{p} entry {i} missing 'sequence'.")
    return checkpoint


def _gather_pt_files(paths: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.pt")))
        elif path.suffix == ".pt":
            files.append(path)
        else:
            raise SchemaError(f"Not a .pt file or directory: {path}")
    if not files:
        raise SchemaError("No .pt files found.")
    return files
