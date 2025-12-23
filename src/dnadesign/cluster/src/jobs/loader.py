"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/jobs/loader.py

Minimal job loader: a job bundles a command ('fit'|'umap'|'analyze') and params.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _package_cluster_dir() -> Path:
    """Return the installed package's cluster directory (…/dnadesign/cluster)."""
    return Path(__file__).resolve().parents[2]


def _nearest_cluster_dir() -> Path | None:
    """
    Walk upward from CWD and return the nearest directory that looks like a
    project-level 'cluster/' folder. Returns None if not found.
    """
    cwd = Path.cwd()
    for base in [cwd, *cwd.parents]:
        if (base / "cluster").exists():
            return base / "cluster"
        if base.name == "cluster":
            return base
    return None


def _search_job_candidates(spec: str | Path) -> List[Path]:
    """
    Build an ordered list of candidate file paths for a job spec.
    The search is explicit and deterministic — we try:
      1) The path exactly as provided (absolute or relative to CWD)
      2) <cluster_root>/<tail> and <cluster_root>/jobs/<tail> for:
         a) $DNADESIGN_CLUSTER_ROOT, if set
         b) nearest project 'cluster/' found by walking up from CWD
         c) the package's own cluster directory (…/src/dnadesign/cluster)

    Where <tail> is the provided spec with a leading 'cluster/' prefix removed
    (so specs like 'cluster/jobs/foo.yaml' work regardless of CWD).
    """
    pspec = Path(spec)
    candidates: List[Path] = []
    # 1) As provided
    candidates.append(pspec if pspec.is_absolute() else (Path.cwd() / pspec))
    # Normalize leading 'cluster/' so we can anchor under known cluster roots
    tail = pspec
    if len(pspec.parts) >= 1 and pspec.parts[0] == "cluster":
        tail = Path(*pspec.parts[1:])
    # Resolve cluster roots (explicit env first, then project, then package)
    roots: List[Path] = []
    env = os.environ.get("DNADESIGN_CLUSTER_ROOT")
    if env:
        roots.append(Path(env).expanduser())
    proj = _nearest_cluster_dir()
    if proj:
        roots.append(proj)
    roots.append(_package_cluster_dir())
    # For each root, try <root>/<tail> and <root>/jobs/<tail> unless tail already starts with 'jobs'
    for root in roots:
        if tail.parts and tail.parts[0] == "jobs":
            candidates.append(root / tail)
        else:
            candidates.append(root / tail)
            candidates.append(root / "jobs" / tail)
    # De-duplicate while preserving order
    seen: set[str] = set()
    out: List[Path] = []
    for c in candidates:
        key = str(c.resolve()) if c.is_absolute() else str(c)
        if key not in seen:
            out.append(c)
            seen.add(key)
    return out


def _normalize_dict_keys(d: Dict[str, Any], *, path: Tuple[str, ...] = ()) -> Dict[str, Any]:
    """
    Recursively normalize mapping keys to snake_case by replacing '-' with '_'.
    Assertively error if both kebab- and snake-case forms coexist at the same level.
    """
    out: Dict[str, Any] = {}
    seen_src: Dict[str, str] = {}
    for k, v in d.items():
        nk = k.replace("-", "_") if isinstance(k, str) else k
        if isinstance(v, dict):
            v = _normalize_dict_keys(v, path=(*path, str(nk)))
        elif isinstance(v, list):
            v = [(_normalize_dict_keys(x, path=(*path, str(nk))) if isinstance(x, dict) else x) for x in v]
        # Collision: e.g., 'x-col' and 'x_col' both present
        if nk in out and seen_src.get(str(nk)) != k:
            here = ".".join(path) if path else "<root>"
            raise ValueError(
                f"Duplicate parameters after key normalization at {here}: "
                f"'{seen_src[str(nk)]}' and '{k}' both map to '{nk}'. "
                "Use only one form (prefer snake_case)."
            )
        out[nk] = v
        seen_src[str(nk)] = k
    return out


def load_job_file(path: str | Path) -> Dict[str, Any]:
    tried: List[str] = []
    for cand in _search_job_candidates(path):
        tried.append(str(cand))
        if cand.exists():
            obj = yaml.safe_load(cand.read_text())
            if not isinstance(obj, dict):
                raise ValueError("Job YAML must be a mapping with keys: command, params.")
            # Normalize keys (kebab-case → snake_case) throughout the job mapping
            obj = _normalize_dict_keys(obj)
            if "params" in obj and not isinstance(obj["params"], dict):
                raise ValueError("Job 'params' must be a mapping.")
            return obj
    lines = [
        f"Job file not found for spec: {path}",
        "Paths tried:",
        *[f"  - {t}" for t in tried[:20]],
    ]
    if len(tried) > 20:
        lines.append("  - …")
    lines.append(
        "Hint: pass an absolute path, or a path relative to your project's 'cluster/' "
        "directory (e.g., 'cluster/jobs/<fit_alias>/umap.yaml'). "
        "You can also set DNADESIGN_CLUSTER_ROOT to point at that 'cluster/' directory."
    )
    raise FileNotFoundError("\n".join(lines))
