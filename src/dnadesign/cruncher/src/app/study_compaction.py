"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/study_compaction.py

Prune transient trial artifacts from Study runs while preserving summary inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.artifacts.layout import elites_path
from dnadesign.cruncher.study.layout import study_manifest_path
from dnadesign.cruncher.study.manifest import load_study_manifest

_TRIAL_DROP_PATHS = (
    Path("optimize") / "tables" / "sequences.parquet",
    Path("optimize") / "tables" / "random_baseline.parquet",
    Path("optimize") / "tables" / "random_baseline_hits.parquet",
    Path("optimize") / "state" / "trace.nc",
    Path("optimize") / "optimizer_move_stats.json.gz",
)


@dataclass(frozen=True)
class StudyCompactionSummary:
    trial_count: int
    candidate_file_count: int
    candidate_bytes: int
    deleted_file_count: int
    deleted_bytes: int


def _trials_root(study_run_dir: Path) -> Path:
    return (study_run_dir / "trials").resolve()


def _require_in_trials_root(path: Path, *, trials_root: Path, label: str) -> None:
    try:
        path.relative_to(trials_root)
    except ValueError as exc:
        raise ValueError(f"{label} must resolve under '{trials_root}': {path}") from exc


def _require_safe_candidate_path(candidate: Path, *, trials_root: Path) -> None:
    if candidate.is_symlink():
        raise ValueError(f"Study compaction candidate must not be a symlink: {candidate}")
    _require_in_trials_root(candidate, trials_root=trials_root, label="Study compaction candidate")
    if not candidate.exists():
        raise FileNotFoundError(f"Study compaction candidate is missing: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Study compaction candidate must be a file: {candidate}")


def _resolve_trial_run_dir(study_run_dir: Path, run_dir: str) -> Path:
    raw_text = str(run_dir).strip()
    if not raw_text:
        raise ValueError("Study trial run_dir must be a non-empty path.")
    raw = Path(raw_text)
    if any(part == ".." for part in raw.parts):
        raise ValueError(f"Study trial run_dir must not contain '..': run_dir={run_dir!r}")
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        resolved = (study_run_dir / raw).resolve()
    trials_root = _trials_root(study_run_dir)
    _require_in_trials_root(
        resolved,
        trials_root=trials_root,
        label=f"Study trial run_dir run_dir={run_dir!r} resolved",
    )
    return resolved


def _candidate_paths(run_dir: Path, *, trials_root: Path) -> list[Path]:
    paths: list[Path] = []
    for relative in _TRIAL_DROP_PATHS:
        candidate = run_dir / relative
        if candidate.exists():
            _require_safe_candidate_path(candidate, trials_root=trials_root)
            paths.append(candidate)
    optimize_tables = run_dir / "optimize" / "tables"
    if optimize_tables.exists():
        for tmp_file in sorted(optimize_tables.glob("*.tmp")):
            if tmp_file.exists():
                _require_safe_candidate_path(tmp_file, trials_root=trials_root)
                paths.append(tmp_file)
    return paths


def compact_study_run(study_run_dir: Path, *, confirm: bool) -> StudyCompactionSummary:
    run_root = study_run_dir.resolve()
    trials_root = _trials_root(run_root)
    manifest = load_study_manifest(study_manifest_path(run_root))

    trial_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    for trial in manifest.trial_runs:
        if trial.run_dir is None:
            if trial.status == "success":
                raise ValueError(
                    f"Successful trial is missing run_dir in study manifest: trial={trial.trial_id} seed={trial.seed}"
                )
            continue
        trial_dir = _resolve_trial_run_dir(run_root, trial.run_dir)
        if trial.status == "success":
            if not trial_dir.exists():
                raise FileNotFoundError(
                    f"Missing trial run directory for successful trial: trial={trial.trial_id} seed={trial.seed} "
                    f"run_dir={trial_dir}"
                )
            if not trial_dir.is_dir():
                raise ValueError(f"Trial run path is not a directory for successful trial: {trial_dir}")
            elite_file = elites_path(trial_dir)
            if not elite_file.exists():
                raise FileNotFoundError(f"Missing elites parquet for successful trial: {elite_file}")
        if trial_dir in seen_dirs:
            continue
        seen_dirs.add(trial_dir)
        trial_dirs.append(trial_dir)

    candidates: list[Path] = []
    for trial_dir in sorted(trial_dirs):
        candidates.extend(_candidate_paths(trial_dir, trials_root=trials_root))
    candidates = sorted(set(candidates))

    candidate_bytes = sum(path.stat().st_size for path in candidates)
    if not confirm:
        return StudyCompactionSummary(
            trial_count=len(trial_dirs),
            candidate_file_count=len(candidates),
            candidate_bytes=int(candidate_bytes),
            deleted_file_count=0,
            deleted_bytes=0,
        )

    deleted_bytes = 0
    deleted_count = 0
    for path in candidates:
        _require_safe_candidate_path(path, trials_root=trials_root)
        size = path.stat().st_size
        path.unlink()
        deleted_count += 1
        deleted_bytes += int(size)
    return StudyCompactionSummary(
        trial_count=len(trial_dirs),
        candidate_file_count=len(candidates),
        candidate_bytes=int(candidate_bytes),
        deleted_file_count=int(deleted_count),
        deleted_bytes=int(deleted_bytes),
    )
