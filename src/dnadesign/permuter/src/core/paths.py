"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/core/paths.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

try:
    # Python 3.9+: importlib.resources.files gives us package install path
    from importlib.resources import files as _pkg_files  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _pkg_files = None
_LOG = logging.getLogger("permuter.paths")


@dataclass(frozen=True)
class JobPaths:
    job_yaml: Path
    job_dir: Path
    refs_csv: Path
    output_root: Path
    dataset_dir: Path
    records_parquet: Path
    ref_fa: Path
    plots_dir: Path


def _expand(s: str, *, job_dir: Path) -> Path:
    # expand ~ and $VARS and ${JOB_DIR}
    s = s or ""
    s = s.replace("${JOB_DIR}", str(job_dir))
    s = os.path.expandvars(s)
    p = Path(os.path.expanduser(s))
    return p if p.is_absolute() else (job_dir / p)


# Public helper: expand a string path using the job directory as the base.
def expand_for_job(value: str | Path, *, job_dir: Path) -> Path:
    """
    Expand ${JOB_DIR}, ~, and $ENV within 'value', and make it absolute
    relative to 'job_dir' when not already absolute.
    """
    return _expand(str(value), job_dir=job_dir).resolve()


def _unique(seq: Iterable[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for p in seq:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _package_jobs_dir() -> Optional[Path]:
    """Installed package jobs dir: .../site-packages/dnadesign/permuter/jobs"""
    try:
        if _pkg_files is None:
            return None
        base = Path(str(_pkg_files("dnadesign.permuter")))
        cand = (base / "jobs").resolve()
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


def candidate_job_dirs() -> List[Path]:
    """
    Ordered search roots for preset YAMLs:
      1) $PERMUTER_JOBS (':'-separated)
      2) CWD and CWD/jobs
      3) repo root (if found) and <root>/src/dnadesign/permuter/jobs
      4) installed package jobs directory
    """
    out: List[Path] = []
    env = os.environ.get("PERMUTER_JOBS", "")
    for chunk in [x for x in env.split(":") if x.strip()]:
        out.append(Path(os.path.expanduser(chunk)).resolve())
    cwd = Path.cwd().resolve()
    out += [cwd, cwd / "jobs"]
    root = _repo_root_from(cwd)
    if root:
        out.append(root)
        out.append(root / "src" / "dnadesign" / "permuter" / "jobs")
    pkg = _package_jobs_dir()
    if pkg:
        out.append(pkg)
    return [p for p in _unique(out) if p.exists()]


def resolve_job_hint(hint: str | Path) -> Path:
    """
    Resolve a job hint that can be:
      • absolute/relative path to YAML
      • bare preset name (we'll search candidate_job_dirs)
    """
    h = Path(str(hint))
    # Direct file path?
    if h.suffix.lower() in (".yaml", ".yml") and h.exists():
        return h.resolve()
    if h.exists():
        return h.resolve()
    # Search as preset name
    base = h.name
    names = [base, f"{base}.yaml", f"{base}.yml"]
    tried: List[Path] = []
    for d in candidate_job_dirs():
        for nm in names:
            cand = (d / nm).resolve()
            tried.append(cand)
            if cand.exists():
                return cand
    # Minimal message by default; detailed list only when DEBUG
    # concise by default; show full tried list only in debug mode or when explicitly asked
    dirs = "\n  - ".join(str(d) for d in candidate_job_dirs())
    show_tried = os.environ.get("PERMUTER_DEBUG_HINTS") == "1" or _LOG.isEnabledFor(
        logging.DEBUG
    )
    if show_tried:
        tried_str = "\n  - ".join(str(p) for p in _unique(tried))
        msg = f"Job YAML '{hint}' not found.\nSearched directories:\n  - {dirs}" + (
            f"\nTried filenames:\n  - {tried_str}" if tried_str else ""
        )
    else:
        msg = (
            f"Job YAML '{hint}' not found.\nSearched directories:\n  - {dirs}\n"
            "Tip: run with -vv for search details or set PERMUTER_DEBUG_HINTS=1."
        )
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


def resolve(
    job_yaml: Path,
    *,
    refs: str,
    output_dir: str,
    ref_name: str,
    out_override: Path | None,
) -> JobPaths:
    job_yaml = Path(job_yaml).expanduser().resolve()
    job_dir = job_yaml.parent

    # Resolve refs CSV relative to YAML location, and assert it's a file
    refs_csv = _expand(refs, job_dir=job_dir).resolve()
    if refs_csv.is_dir():
        example = refs_csv / "refs.csv"
        raise IsADirectoryError(
            "Refs path points to a directory; a CSV file is required.\n"
            f"Given: {refs_csv}\n"
            f"Hint: set job.input.refs to the CSV (e.g., {example})"
        )
    if not refs_csv.exists():
        raise FileNotFoundError(f"Refs CSV not found: {refs_csv}")

    # Determine output root (strict: no silent fallbacks)
    output_root = (
        Path(out_override).expanduser().resolve()
        if out_override is not None
        else _expand(output_dir, job_dir=job_dir).resolve()
    )
    if not _is_writable_dir(output_root):
        raise PermissionError(
            f"Output root not writable: {output_root}. "
            "Use --out or set $PERMUTER_OUTPUT_ROOT to a writable location."
        )

    # Dataset directory
    # Layouts:
    #   nested      → <output_root>/<ref>/
    #   flat        → <output_root>/                      (one dataset per job)
    ref_dir = ref_name or "__PENDING__"
    layout = os.environ.get("PERMUTER_LAYOUT", "").strip().lower()
    if not layout:
        if os.environ.get("PERMUTER_FLAT_RESULTS", "") in ("1", "true", "True"):
            layout = "flat_jobref"

    if layout in ("flat", "flat_job", "job"):
        dataset_dir = output_root
    elif layout in ("flat_jobref", "flat_ref", "jobref"):
        base = output_root.parent
        dataset_dir = (base / f"{output_root.name}__{ref_dir}").resolve()
    else:  # "nested" (default) or unknown → nested
        dataset_dir = (output_root / ref_dir).resolve()
    records_parquet = dataset_dir / "records.parquet"
    ref_fa = dataset_dir / "REF.fa"
    plots_dir = dataset_dir / "plots"

    return JobPaths(
        job_yaml=job_yaml,
        job_dir=job_dir,
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
        "csv", "tsv", "parquet", "pqt", "json", "jsonl", "yaml", "yml",
        "fa", "fasta", "txt", "gz", "bz2", "xz", "zip"
    }:
        return True
    if key_hint:
        k = key_hint.lower()
        if k.endswith(("_path", "_file", "_dir")):
            return True
        if k in {"from_dataset", "dataset", "codon_table", "refs"}:
            return True
    return False

def expand_param_paths(params: dict | None, *, job_dir: Path) -> dict:
    """
    Deep-copy and expand any string values in a params mapping that appear to be paths.
    Uses job_dir to resolve ${JOB_DIR}, $ENV, and ~, and to make relative paths absolute.
    """
    def _map(v: Any, key_hint: Optional[str] = None) -> Any:
        if isinstance(v, str) and _looks_pathlike(v, key_hint=key_hint):
            try:
                p = _expand(v, job_dir=job_dir).resolve()
                if _LOG.isEnabledFor(logging.DEBUG):
                    _LOG.debug("expand_param_paths: %r → %s", v, p)
                return str(p)
            except Exception:
                # Leave unchanged; concrete validators will raise clear errors.
                return v
        if isinstance(v, list):
            return [_map(x, None) for x in v]
        if isinstance(v, dict):
            return {kk: _map(vv, kk) for kk, vv in v.items()}
        return v
    return {k: _map(v, k) for k, v in (params or {}).items()}