"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/core/paths.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from importlib.resources import files as _pkg_files  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _pkg_files = None

CONFIG_NAME = "config.yaml"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_RESOURCE_DIR = _PACKAGE_ROOT / "src" / "resources"
_FLAT_LAYOUTS = {"flat"}
_NESTED_LAYOUTS = {"nested"}
_ALLOWED_LAYOUTS = _FLAT_LAYOUTS | _NESTED_LAYOUTS


@dataclass(frozen=True)
class WorkspacePaths:
    config_yaml: Path
    workspace_dir: Path
    refs_csv: Path
    output_root: Path
    dataset_dir: Path
    records_parquet: Path
    ref_fa: Path
    plots_dir: Path


def _workspaces_dir(workspace_dir: Path) -> Path:
    if workspace_dir.parent.name == "workspaces":
        return workspace_dir.parent.resolve()
    return workspace_dir.parent.resolve()


def _expand(s: str, *, workspace_dir: Path) -> Path:
    s = s or ""
    if "${JOB_DIR}" in s:
        raise ValueError("${JOB_DIR} is not supported; use ${WORKSPACE_DIR} in Permuter workspace configs")
    if "${PACKAGE_ROOT}" in s:
        raise ValueError(
            "${PACKAGE_ROOT} is not supported in Permuter workspace configs; "
            "use ${WORKSPACE_DIR}, ${WORKSPACES_DIR}, or ${PERMUTER_RESOURCE_DIR}"
        )
    s = s.replace("${WORKSPACE_DIR}", str(workspace_dir))
    s = s.replace("${WORKSPACES_DIR}", str(_workspaces_dir(workspace_dir)))
    s = s.replace("${PERMUTER_RESOURCE_DIR}", str(_RESOURCE_DIR))
    s = os.path.expandvars(s)
    p = Path(os.path.expanduser(s))
    return p if p.is_absolute() else (workspace_dir / p)


def expand_for_workspace(value: str | Path, *, workspace_dir: Path) -> Path:
    return _expand(str(value), workspace_dir=workspace_dir).resolve()


def _unique(seq: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in seq:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _package_workspaces_dir() -> Optional[Path]:
    try:
        if _pkg_files is None:
            return None
        base = Path(str(_pkg_files("dnadesign.permuter")))
        cand = (base / "workspaces").resolve()
        return cand if cand.exists() else None
    except Exception:
        return None


def _repo_root_from(start: Path) -> Optional[Path]:
    """Walk upward to a plausible repo root (pyproject.toml or src/dnadesign/permuter present)."""
    cur = start.resolve()
    for _ in range(12):
        if (cur / "pyproject.toml").exists():
            return cur
        if (cur / "src" / "dnadesign" / "permuter").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def candidate_workspace_roots() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("PERMUTER_WORKSPACES", "")
    for chunk in [x for x in env.split(":") if x.strip()]:
        out.append(Path(os.path.expanduser(chunk)).resolve())
    cwd = Path.cwd().resolve()
    out += [cwd, cwd / "workspaces"]
    root = _repo_root_from(cwd)
    if root:
        out.append(root / "src" / "dnadesign" / "permuter" / "workspaces")
    pkg = _package_workspaces_dir()
    if pkg:
        out.append(pkg)
    return [p for p in _unique(out) if p.exists()]


def resolve_workspace_config_hint(hint: str | Path) -> Path:
    h = Path(str(hint))
    if h.exists():
        resolved = h.resolve()
        if resolved.is_dir():
            config = resolved / CONFIG_NAME
            if config.exists():
                return config
            raise FileNotFoundError(f"Workspace directory does not contain {CONFIG_NAME}: {resolved}")
        if resolved.name != CONFIG_NAME:
            raise ValueError(f"Permuter workspace config must be named {CONFIG_NAME!r}: {resolved}")
        return resolved

    base = h.name
    tried: list[Path] = []
    for root in candidate_workspace_roots():
        cand = (root / base / CONFIG_NAME).resolve()
        tried.append(cand)
        if cand.exists():
            return cand
    dirs = "\n  - ".join(str(d) for d in candidate_workspace_roots())
    tried_str = "\n  - ".join(str(p) for p in _unique(tried))
    msg = f"Workspace '{hint}' not found.\nSearched workspace roots:\n  - {dirs}\nTried config paths:\n  - {tried_str}"
    raise FileNotFoundError(msg)


def normalize_data_path(p: Path | str) -> Path:
    """
    Accept either a dataset directory or records.parquet file.
    Returns a Path to records.parquet.
    """
    # Expand env vars and ~ regardless of caller's CWD.
    raw = os.path.expandvars(str(p) or "")
    path = Path(os.path.expanduser(raw)).resolve()
    if path.is_dir():
        return (path / "records.parquet").resolve()
    return path


def _is_writable_dir(p: Path) -> bool:
    try:
        p = p.resolve()
        p.mkdir(parents=True, exist_ok=True)
        test = p / ".permute_write_test"
        test.write_text("", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _configured_output_root(
    *,
    output_dir: str,
    workspace_dir: Path,
    out_override: Path | None,
) -> Path:
    if out_override is not None:
        return Path(out_override).expanduser().resolve()
    configured = _expand(output_dir, workspace_dir=workspace_dir).resolve()
    env_root = os.environ.get("PERMUTER_OUTPUT_ROOT", "").strip()
    if not env_root:
        return configured
    scope = workspace_dir.name
    if not scope:
        raise ValueError(f"Cannot derive PERMUTER_OUTPUT_ROOT child name from output.dir={output_dir!r}")
    return (Path(os.path.expandvars(env_root)).expanduser().resolve() / scope).resolve()


def _normalize_layout(layout: str | None) -> str:
    source = "output.layout" if layout else "PERMUTER_LAYOUT"
    raw = str(layout or os.environ.get("PERMUTER_LAYOUT", "")).strip().lower()
    if not raw:
        return "flat"
    if raw not in _ALLOWED_LAYOUTS:
        raise ValueError(f"Invalid {source} {raw!r}. Allowed: {sorted(_ALLOWED_LAYOUTS)}")
    return raw


def resolve(
    config_yaml: Path,
    *,
    refs: str,
    output_dir: str,
    ref_name: str,
    out_override: Path | None,
    layout: str | None = None,
    require_writable_output: bool = True,
) -> WorkspacePaths:
    config_yaml = Path(config_yaml).expanduser().resolve()
    if config_yaml.name != CONFIG_NAME:
        raise ValueError(f"Permuter workspace config must be named {CONFIG_NAME!r}: {config_yaml}")
    workspace_dir = config_yaml.parent

    refs_csv = _expand(refs, workspace_dir=workspace_dir).resolve()
    if refs_csv.is_dir():
        example = refs_csv / "refs.csv"
        raise IsADirectoryError(
            "Refs path points to a directory; a CSV file is required.\n"
            f"Given: {refs_csv}\n"
            f"Hint: set scope.input.refs to the CSV (e.g., {example})"
        )
    if not refs_csv.exists():
        raise FileNotFoundError(f"Refs CSV not found: {refs_csv}")

    output_root = _configured_output_root(
        output_dir=output_dir,
        workspace_dir=workspace_dir,
        out_override=out_override,
    )
    if require_writable_output and not _is_writable_dir(output_root):
        raise PermissionError(
            f"Output root not writable: {output_root}. Use --out or set $PERMUTER_OUTPUT_ROOT to a writable location."
        )

    ref_dir = ref_name or "__PENDING__"
    layout = _normalize_layout(layout)

    if layout in _FLAT_LAYOUTS:
        dataset_dir = output_root
    else:
        dataset_dir = (output_root / ref_dir).resolve()
    records_parquet = dataset_dir / "records.parquet"
    ref_fa = dataset_dir / "REF.fa"
    plots_dir = dataset_dir / "plots"

    return WorkspacePaths(
        config_yaml=config_yaml,
        workspace_dir=workspace_dir,
        refs_csv=refs_csv,
        output_root=output_root,
        dataset_dir=dataset_dir,
        records_parquet=records_parquet,
        ref_fa=ref_fa,
        plots_dir=plots_dir,
    )


def _looks_pathlike(s: str, *, key_hint: Optional[str] = None) -> bool:
    """
    Conservative heuristic: only treat as a path when it *looks* like one.
    """
    if not s:
        return False
    s2 = s.strip()
    if any(tok in s2 for tok in ("/", "\\", "~", "${")):
        return True
    ext = s2.lower().rsplit(".", 1)[-1] if "." in s2 else ""
    if ext in {
        "csv",
        "tsv",
        "parquet",
        "pqt",
        "json",
        "jsonl",
        "yaml",
        "yml",
        "fa",
        "fasta",
        "txt",
        "gz",
        "bz2",
        "xz",
        "zip",
    }:
        return True
    if key_hint:
        k = key_hint.lower()
        if k.endswith(("_path", "_file", "_dir")):
            return True
        if k in {"from_dataset", "dataset", "codon_table", "refs"}:
            return True
    return False


def expand_param_paths(params: dict | None, *, workspace_dir: Path) -> dict:
    """Deep-copy and expand string values that look like paths."""

    def _map(v: Any, key_hint: Optional[str] = None) -> Any:
        if isinstance(v, str) and _looks_pathlike(v, key_hint=key_hint):
            p = _expand(v, workspace_dir=workspace_dir).resolve()
            return str(p)
        if isinstance(v, list):
            return [_map(x, None) for x in v]
        if isinstance(v, dict):
            return {kk: _map(vv, kk) for kk, vv in v.items()}
        return v

    return {k: _map(v, k) for k, v in (params or {}).items()}
