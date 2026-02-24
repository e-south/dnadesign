"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/study/helpers.py

Shared Study workflow helpers for target selection, trial expansion, and run resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from dnadesign.cruncher.artifacts.layout import RUN_META_DIR, manifest_path
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.study.identity import (
    compute_study_id as _compute_study_id,
)
from dnadesign.cruncher.study.identity import (
    compute_study_spec_hash as _spec_hash,
)
from dnadesign.cruncher.study.identity import (
    study_spec_payload as _spec_payload,
)
from dnadesign.cruncher.study.manifest import StudyTrialRun
from dnadesign.cruncher.study.overrides import extract_factor_columns
from dnadesign.cruncher.study.schema_models import StudySpec, StudyTrial

_MAX_EXPANDED_STUDY_TRIALS = 500


@dataclass(frozen=True)
class _TargetSet:
    index: int
    tfs: tuple[str, ...]


def _trial_key(trial_run: StudyTrialRun) -> tuple[str, int, int]:
    return (str(trial_run.trial_id), int(trial_run.seed), int(trial_run.target_set_index))


def _config_with_regulator_set(cfg: CruncherConfig, tfs: tuple[str, ...]) -> CruncherConfig:
    payload = cfg.model_dump(mode="python")
    workspace = payload.get("workspace")
    if not isinstance(workspace, dict):
        raise ValueError("Config missing workspace mapping.")
    workspace["regulator_sets"] = [list(tfs)]
    payload["workspace"] = workspace
    return CruncherConfig.model_validate(payload)


def _config_with_regulator_sets(cfg: CruncherConfig, target_sets: list[_TargetSet]) -> CruncherConfig:
    payload = cfg.model_dump(mode="python")
    workspace = payload.get("workspace")
    if not isinstance(workspace, dict):
        raise ValueError("Config missing workspace mapping.")
    workspace["regulator_sets"] = [list(item.tfs) for item in sorted(target_sets, key=lambda item: item.index)]
    payload["workspace"] = workspace
    return CruncherConfig.model_validate(payload)


def _config_with_out_dir(cfg: CruncherConfig, *, out_dir_relative: str) -> CruncherConfig:
    payload = cfg.model_dump(mode="python")
    workspace = payload.get("workspace")
    if not isinstance(workspace, dict):
        raise ValueError("Config missing workspace mapping.")
    workspace["out_dir"] = out_dir_relative
    payload["workspace"] = workspace
    return CruncherConfig.model_validate(payload)


def _resolve_target_sets(spec: StudySpec, cfg: CruncherConfig) -> list[_TargetSet]:
    target = spec.target
    if getattr(target, "kind", None) == "regulator_set":
        set_index = int(target.set_index)
        if set_index < 1 or set_index > len(cfg.regulator_sets):
            raise ValueError(
                f"study.target.set_index={set_index} is out of range "
                f"(workspace has {len(cfg.regulator_sets)} regulator_sets)"
            )
        selected = tuple(str(tf) for tf in cfg.regulator_sets[set_index - 1])
        if not selected:
            raise ValueError(f"study.target.set_index={set_index} resolves to an empty regulator set.")
        return [_TargetSet(index=set_index, tfs=selected)]
    raise ValueError(f"Unsupported study target kind: {getattr(target, 'kind', None)!r}")


def _expand_trials(spec: StudySpec) -> list[StudyTrial]:
    trial_rows: list[StudyTrial] = list(spec.trials)
    if len(trial_rows) > _MAX_EXPANDED_STUDY_TRIALS:
        raise ValueError(f"Study defines too many trials (> {_MAX_EXPANDED_STUDY_TRIALS}).")
    generated_ids: set[str] = {item.id for item in trial_rows}
    for grid in spec.trial_grids:
        axis_paths = sorted(grid.factors.keys())
        axis_values = [grid.factors[path] for path in axis_paths]
        for idx, combo_values in enumerate(product(*axis_values), start=1):
            trial_id = f"{grid.id_prefix}_{idx}"
            if trial_id in generated_ids:
                raise ValueError(f"Duplicate trial id after grid expansion: {trial_id}")
            generated_ids.add(trial_id)
            trial_rows.append(
                StudyTrial(
                    id=trial_id,
                    factors={path: combo_values[path_idx] for path_idx, path in enumerate(axis_paths)},
                )
            )
            if len(trial_rows) > _MAX_EXPANDED_STUDY_TRIALS:
                raise ValueError(f"Study defines too many expanded trials (> {_MAX_EXPANDED_STUDY_TRIALS}).")
    return trial_rows


def _dotpath_value(payload: dict[str, Any], path: str) -> Any:
    cursor: Any = payload
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise ValueError(f"Unsupported study factor path in base config: {path}")
        cursor = cursor[part]
    return cursor


def _canonical(value: Any) -> str:
    import json

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Study factor value must be JSON-serializable and finite: {value!r}") from exc


def validate_study_factor_alignment(*, cfg: CruncherConfig, spec: StudySpec, trials: list[StudyTrial]) -> None:
    payload = cfg.model_dump(mode="python")
    factor_keys = sorted({key for trial in trials for key in trial.factors.keys()})
    for key in factor_keys:
        base_value = _dotpath_value(payload, key)
        base_encoded = _canonical(base_value)
        encoded_values = {_canonical(trial.factors.get(key, base_value)) for trial in trials}
        if base_encoded not in encoded_values:
            raise ValueError(f"Study factor '{key}' must include the base config value ({base_value!r}) across trials.")
    if bool(spec.replays.mmr_sweep.enabled):
        base_diversity = float(_dotpath_value(payload, "sample.elites.select.diversity"))
        replay_values = {float(item) for item in spec.replays.mmr_sweep.diversity_values}
        if base_diversity not in replay_values:
            raise ValueError(
                "study.replays.mmr_sweep replay diversity_values must include base config diversity "
                f"({base_diversity})."
            )


def _resolve_trial_runs(
    *,
    trials: list[StudyTrial],
    seeds: list[int],
    target_sets: list[_TargetSet],
) -> list[StudyTrialRun]:
    rows: list[StudyTrialRun] = []
    for trial in sorted(trials, key=lambda item: item.id):
        for seed in sorted(seeds):
            for target_set in sorted(target_sets, key=lambda item: item.index):
                rows.append(
                    StudyTrialRun(
                        trial_id=trial.id,
                        seed=int(seed),
                        target_set_index=int(target_set.index),
                        target_tfs=list(target_set.tfs),
                        factors=dict(trial.factors),
                        factor_columns=extract_factor_columns(trial.factors),
                    )
                )
    return rows


def _discover_run_dir(seed_root: Path) -> Path:
    manifests = sorted(seed_root.rglob("run_manifest.json"))
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for item in manifests:
        run_dir = item.parent
        if run_dir.name == RUN_META_DIR:
            run_dir = run_dir.parent
        resolved = run_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        run_dirs.append(run_dir)
    if not run_dirs:
        raise FileNotFoundError(f"Study trial produced no run_manifest.json under {seed_root}")
    if len(run_dirs) > 1:
        rendered = ", ".join(str(path) for path in run_dirs)
        raise ValueError(f"Study trial expected exactly one run directory but found {len(run_dirs)}: {rendered}")
    if not manifest_path(run_dirs[0]).exists():
        raise FileNotFoundError(f"Study trial run is missing run_manifest.json: {run_dirs[0]}")
    return run_dirs[0]


__all__ = [
    "_TargetSet",
    "_compute_study_id",
    "_config_with_out_dir",
    "_config_with_regulator_set",
    "_config_with_regulator_sets",
    "_discover_run_dir",
    "_expand_trials",
    "validate_study_factor_alignment",
    "_resolve_target_sets",
    "_resolve_trial_runs",
    "_spec_hash",
    "_spec_payload",
    "_trial_key",
]
