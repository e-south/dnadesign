# ABOUTME: Core formatting helpers for OPAL CLI output.
# ABOUTME: Provides markup-aware formatting utilities and shared helpers.
"""Core formatting helpers for OPAL CLI."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Mapping

# -------------
# Markup gate
# -------------
_TRUTHY = {"1", "true", "yes", "on"}


def _markup_enabled() -> bool:
    val = os.getenv("OPAL_CLI_MARKUP", "").strip().lower()
    if val == "":
        return True
    return val in _TRUTHY


def _b(s: str) -> str:
    return f"[bold]{s}[/]" if _markup_enabled() else s


def _t(s: str) -> str:
    return f"[bold cyan]{s}[/]" if _markup_enabled() else s


def _dim(s: str) -> str:
    return f"[dim]{s}[/]" if _markup_enabled() else s


# -----------------------
# Core formatting helpers
# -----------------------


def _indent(s: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in s.splitlines())


def _fmt_multiline(v: object) -> str:
    s = v if isinstance(v, str) else str(v)
    return ("\n" + _indent(s, 2)) if "\n" in s else s


def kv_block(title: str, items: Mapping[str, object]) -> str:
    lines = [_t(str(title))]
    for k in items.keys():
        v = _fmt_multiline(items[k])
        key = f"{_b(str(k))}"
        lines.append(f"  {key:24s}: {v}")
    return "\n".join(lines)


def bullet_list(title: str, rows: Iterable[str]) -> str:
    rows = [str(r) for r in rows]
    bullet = "•"
    if not rows:
        return f"{_t(title)}\n  {bullet} {_dim('(none)')}"
    return _t(title) + "\n  " + f"{bullet} " + f"\n  {bullet} ".join(rows)


def short_array(a, maxlen: int = 8) -> str:
    try:
        import numpy as np  # optional

        arr = np.asarray(a).ravel().tolist()
    except Exception:
        try:
            arr = list(a)
        except Exception:
            return str(a)
    if len(arr) <= maxlen:
        return str(arr)
    head = ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in arr[:maxlen])
    return f"[{head}, …] (len={len(arr)})"


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": str(obj)}


def _fmt_params(params: Any) -> str:
    try:
        txt = json.dumps(params, indent=2, sort_keys=True)
        return txt if not _markup_enabled() else f"[white]{txt}[/]"
    except Exception:
        return str(params)


def _truncate(s: str, n: int = 48) -> str:
    s = str(s)
    return s if len(s) <= n else (s[: n - 1] + "…")


def _sha_short(sha: str, n: int = 12) -> str:
    sha = (sha or "").strip()
    return sha[:n] if sha else ""
