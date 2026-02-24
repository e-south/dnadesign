"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/study_workflow.py

Orchestrate first-class Study runs, replays, and aggregate reporting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import json
import logging
import multiprocessing
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.cruncher.analysis.mmr_sweep_service import run_mmr_sweep_for_run
from dnadesign.cruncher.app.sample.resources import _load_pwms_for_set, _lockmap_for
from dnadesign.cruncher.app.sample_workflow import run_sample
from dnadesign.cruncher.app.study.helpers import (
    _compute_study_id,
    _config_with_out_dir,
    _config_with_regulator_set,
    _config_with_regulator_sets,
    _discover_run_dir,
    _expand_trials,
    _resolve_target_sets,
    _resolve_trial_runs,
    _spec_hash,
    _spec_payload,
    _TargetSet,
    _trial_key,
    validate_study_factor_alignment,
)
from dnadesign.cruncher.app.study_summary import summarize_study_run
from dnadesign.cruncher.app.target_service import has_blocking_target_errors, target_statuses
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_yaml
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.core.labels import regulator_sets
from dnadesign.cruncher.study.layout import (
    resolve_study_run_dir,
    spec_frozen_path,
    study_log_path,
    study_manifest_path,
    study_plot_glob,
    study_plots_dir,
    study_status_path,
    study_tables_dir,
    trial_run_pointer_path,
    trial_seed_root,
)
from dnadesign.cruncher.study.load import load_study_spec
from dnadesign.cruncher.study.manifest import (
    StudyManifestV1,
    StudyStatusV1,
    StudyTrialRun,
    load_study_manifest,
    load_study_status,
    summarize_trial_statuses,
    utc_now_iso,
    write_study_manifest,
    write_study_status,
)
from dnadesign.cruncher.study.overrides import apply_dotpath_overrides
from dnadesign.cruncher.study.schema_models import StudySpec, StudyTrial
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.utils.paths import resolve_lock_path, resolve_workspace_root

_PROFILE_OUTPUT_OVERRIDES: dict[str, dict[str, Any]] = {
    "minimal": {
        "sample.output.save_trace": False,
        "sample.output.save_random_baseline": False,
        "sample.output.save_optimizer_move_stats": False,
        "sample.output.include_tune_in_sequences": False,
        "sample.output.live_metrics": False,
    },
    "analysis_ready": {},
}
logger = logging.getLogger(__name__)


def _preflight_trial_config_contracts(
    spec: StudySpec,
    base_cfg: CruncherConfig,
    *,
    trials: list[StudyTrial],
    target_sets: list[_TargetSet],
) -> None:
    profile = str(spec.artifacts.trial_output_profile)
    require_mmr_sequences = bool(spec.replays.mmr_sweep.enabled)
    for trial in sorted(trials, key=lambda item: item.id):
        for target_set in sorted(target_sets, key=lambda item: item.index):
            targeted_cfg = _config_with_regulator_set(base_cfg, target_set.tfs)
            try:
                cfg_with_trial = apply_dotpath_overrides(targeted_cfg, trial.factors)
            except Exception as exc:
                raise ValueError(
                    f"Study preflight trial config failed for trial={trial.id} target_set={target_set.index}: {exc}"
                ) from exc
            for seed in sorted(spec.replicates.seeds):
                try:
                    cfg_with_seed = apply_dotpath_overrides(cfg_with_trial, {spec.replicates.seed_path: int(seed)})
                    cfg_with_profile = _apply_profile_overrides(cfg_with_seed, profile)
                except Exception as exc:
                    raise ValueError(
                        "Study preflight trial config failed for "
                        f"trial={trial.id} seed={int(seed)} target_set={target_set.index}: {exc}"
                    ) from exc
                if require_mmr_sequences:
                    sample_cfg = cfg_with_profile.sample
                    if sample_cfg is None or not bool(sample_cfg.output.save_sequences):
                        raise ValueError(
                            "Study preflight trial config failed for "
                            f"trial={trial.id} seed={int(seed)} target_set={target_set.index}: "
                            "study.replays.mmr_sweep requires sample.output.save_sequences=true for every "
                            "trial/profile combination."
                        )


def _apply_profile_overrides(cfg: CruncherConfig, profile: str) -> CruncherConfig:
    mapping = _PROFILE_OUTPUT_OVERRIDES.get(profile)
    if mapping is None:
        raise ValueError(f"Unsupported study artifact profile: {profile}")
    return apply_dotpath_overrides(cfg, mapping)


def _ensure_lock_parse_and_targets_ready(cfg: CruncherConfig, config_path: Path) -> None:
    lock_path = resolve_lock_path(config_path)
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing lockfile for study run: {lock_path}. Run `cruncher lock` first.")
    statuses = target_statuses(cfg=cfg, config_path=config_path)
    if has_blocking_target_errors(statuses):
        details = "; ".join(
            f"{status.tf_name}:{status.status}" for status in statuses if status.status not in {"ready", "warning"}
        )
        raise ValueError(
            f"Study preflight target readiness failed: {details}. Run `cruncher targets status -c {config_path}`."
        )
    if cfg.sample is None:
        raise ValueError("Study preflight requires sample configuration.")
    try:
        lockmap = _lockmap_for(cfg, config_path)
    except Exception as exc:
        raise ValueError(
            f"Study preflight parse readiness failed while validating lock/catalog artifacts: {exc}. "
            "Run `cruncher parse -c <config>` and fix parse/catalog artifacts before study execution."
        ) from exc
    validated_tfs: set[str] = set()
    for set_index, group in enumerate(regulator_sets(cfg.regulator_sets), start=1):
        if not group:
            raise ValueError(f"Study preflight parse readiness failed: regulator set {set_index} is empty.")
        seen: set[str] = set()
        tfs = [tf for tf in group if not (tf in seen or seen.add(tf))]
        missing_tfs = sorted(tf for tf in tfs if tf not in validated_tfs)
        if not missing_tfs:
            continue
        try:
            _load_pwms_for_set(
                cfg=cfg,
                config_path=config_path,
                tfs=missing_tfs,
                lockmap=lockmap,
            )
            validated_tfs.update(missing_tfs)
        except Exception as exc:
            raise ValueError(
                f"Study preflight parse readiness failed for set {set_index}: {exc}. "
                "Run `cruncher parse -c <config>` and fix parse/catalog artifacts before study execution."
            ) from exc


def _merge_resume_state(expected: list[StudyTrialRun], existing: list[StudyTrialRun]) -> list[StudyTrialRun]:
    existing_by_key = {_trial_key(item): item for item in existing}
    merged: list[StudyTrialRun] = []
    for item in expected:
        key = _trial_key(item)
        existing_item = existing_by_key.get(key)
        if existing_item is None:
            merged.append(item)
            continue
        if existing_item.status == "running":
            existing_item.status = "pending"
            existing_item.error = None
            existing_item.started_at = None
            existing_item.finished_at = None
        merged.append(existing_item)
    expected_keys = {_trial_key(item) for item in expected}
    existing_keys = set(existing_by_key.keys())
    if expected_keys != existing_keys:
        missing = sorted(expected_keys - existing_keys)
        extra = sorted(existing_keys - expected_keys)
        raise ValueError(
            f"Study resume mismatch between current spec and existing manifest (missing={missing}, extra={extra})."
        )
    return merged


def _refresh_status(
    status: StudyStatusV1,
    manifest: StudyManifestV1,
    *,
    final: bool = False,
    failed: bool = False,
) -> None:
    counts = summarize_trial_statuses(manifest.trial_runs)
    status.total_runs = int(counts["total_runs"])
    status.pending_runs = int(counts["pending_runs"])
    status.running_runs = int(counts["running_runs"])
    status.success_runs = int(counts["success_runs"])
    status.error_runs = int(counts["error_runs"])
    status.skipped_runs = int(counts["skipped_runs"])
    if failed:
        status.status = "failed"
    elif status.running_runs > 0:
        status.status = "running"
    elif status.error_runs > 0:
        status.status = "completed_with_errors"
    else:
        status.status = "completed"
    status.updated_at = utc_now_iso()
    if final:
        status.finished_at = utc_now_iso()


def _append_log(study_run_dir: Path, message: str) -> None:
    log_path = study_log_path(study_run_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _run_one_trial(
    *,
    base_cfg: CruncherConfig,
    base_config_path: Path,
    workspace_root: Path,
    spec: StudySpec,
    study_run_dir: Path,
    trial_run: StudyTrialRun,
    progress_bar: bool,
    quiet_logs: bool,
) -> Path:
    return _run_one_trial_from_payload(
        base_cfg=base_cfg,
        base_config_path=base_config_path,
        workspace_root=workspace_root,
        study_run_dir=study_run_dir,
        trial_run=trial_run,
        seed_path=str(spec.replicates.seed_path),
        profile=str(spec.artifacts.trial_output_profile),
        progress_bar=progress_bar,
        quiet_logs=quiet_logs,
    )


def _run_one_trial_from_payload(
    *,
    base_cfg: CruncherConfig,
    base_config_path: Path,
    workspace_root: Path,
    study_run_dir: Path,
    trial_run: StudyTrialRun,
    seed_path: str,
    profile: str,
    progress_bar: bool,
    quiet_logs: bool,
) -> Path:
    targeted_cfg = _config_with_regulator_set(base_cfg, tuple(trial_run.target_tfs))
    cfg_with_trial = apply_dotpath_overrides(targeted_cfg, trial_run.factors)
    cfg_with_seed = apply_dotpath_overrides(cfg_with_trial, {seed_path: int(trial_run.seed)})
    cfg_with_profile = _apply_profile_overrides(cfg_with_seed, profile)

    seed_root = trial_seed_root(study_run_dir, trial_id=trial_run.trial_id, seed=trial_run.seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    out_dir_relative = seed_root.resolve().relative_to(workspace_root.resolve()).as_posix()
    cfg_final = _config_with_out_dir(cfg_with_profile, out_dir_relative=out_dir_relative)

    previous_disable = logging.root.manager.disable
    if quiet_logs:
        logging.disable(logging.INFO)
    try:
        run_sample(
            cfg_final,
            base_config_path,
            force_overwrite=True,
            progress_bar=progress_bar,
            progress_every=0,
            register_run_in_index=False,
        )
    finally:
        if quiet_logs:
            logging.disable(previous_disable)
    run_dir = _discover_run_dir(seed_root)
    pointer = trial_run_pointer_path(study_run_dir, trial_id=trial_run.trial_id, seed=trial_run.seed)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(run_dir.resolve()))
    return run_dir.resolve()


def _run_one_trial_worker(
    *,
    base_config_path: str,
    workspace_root: str,
    study_run_dir: str,
    trial_run_payload: dict[str, Any],
    seed_path: str,
    profile: str,
    progress_bar: bool,
    quiet_logs: bool,
) -> str:
    trial_run = StudyTrialRun.model_validate(trial_run_payload)
    cfg = load_config(Path(base_config_path))
    run_dir = _run_one_trial_from_payload(
        base_cfg=cfg,
        base_config_path=Path(base_config_path),
        workspace_root=Path(workspace_root),
        study_run_dir=Path(study_run_dir),
        trial_run=trial_run,
        seed_path=seed_path,
        profile=profile,
        progress_bar=progress_bar,
        quiet_logs=quiet_logs,
    )
    return str(run_dir.resolve())


@dataclass(frozen=True)
class _TrialWorkerProcess:
    process: multiprocessing.Process
    result_path: Path
    trial_index: int


def _trial_worker_result_path(*, study_run_dir: Path, trial_run: StudyTrialRun) -> Path:
    return trial_seed_root(study_run_dir, trial_id=trial_run.trial_id, seed=trial_run.seed) / "worker_result.json"


def _run_one_trial_worker_process_entry(
    *,
    base_config_path: str,
    workspace_root: str,
    study_run_dir: str,
    trial_run_payload: dict[str, Any],
    seed_path: str,
    profile: str,
    progress_bar: bool,
    quiet_logs: bool,
    result_path: str,
) -> None:
    result_file = Path(result_path)
    payload: dict[str, object]
    try:
        run_dir = _run_one_trial_worker(
            base_config_path=base_config_path,
            workspace_root=workspace_root,
            study_run_dir=study_run_dir,
            trial_run_payload=trial_run_payload,
            seed_path=seed_path,
            profile=profile,
            progress_bar=progress_bar,
            quiet_logs=quiet_logs,
        )
        payload = {"ok": True, "run_dir": str(Path(run_dir).resolve())}
    except Exception as exc:
        payload = {"ok": False, "error": str(exc)}
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps(payload), encoding="utf-8")


def _load_worker_result(result_path: Path, *, exitcode: int | None) -> tuple[bool, str]:
    if not result_path.exists():
        if exitcode is None:
            raise RuntimeError(f"Study trial worker did not emit result file: {result_path}")
        raise RuntimeError(
            f"Study trial worker exited with code {int(exitcode)} and emitted no result file: {result_path}"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Study trial worker result payload must be a JSON object: {result_path}")
    ok_value = payload.get("ok")
    if not isinstance(ok_value, bool):
        raise RuntimeError(f"Study trial worker result missing boolean 'ok' field: {result_path}")
    if ok_value:
        run_dir = payload.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir.strip():
            raise RuntimeError(f"Study trial worker success payload missing run_dir: {result_path}")
        return True, str(Path(run_dir).resolve())
    error = payload.get("error")
    if not isinstance(error, str) or not error.strip():
        if exitcode is None:
            raise RuntimeError(f"Study trial worker failed with no error payload: {result_path}")
        raise RuntimeError(f"Study trial worker exited with code {int(exitcode)} and no error payload: {result_path}")
    return False, error


def _mark_pending_as_skipped(manifest: StudyManifestV1, *, reason: str) -> None:
    for idx, trial_run in enumerate(manifest.trial_runs):
        if trial_run.status == "pending":
            trial_run.status = "skipped"
            trial_run.error = reason
            trial_run.finished_at = utc_now_iso()
            manifest.trial_runs[idx] = trial_run


@dataclass(frozen=True)
class _StudyBootstrap:
    resolved_spec_path: Path
    spec: StudySpec
    base_config_path: Path
    workspace_root: Path
    base_cfg: CruncherConfig
    expanded_trials: list[StudyTrial]
    target_sets: list[_TargetSet]
    base_config_sha: str
    spec_sha: str
    study_id: str
    study_run_dir: Path
    manifest_file: Path
    status_file: Path


def _bootstrap_study_inputs(spec_path: Path) -> _StudyBootstrap:
    resolved_spec_path = spec_path.resolve()
    spec = load_study_spec(resolved_spec_path)
    base_config_path = spec.base_config.resolve()
    workspace_root = resolve_workspace_root(base_config_path)
    base_cfg = load_config(base_config_path)
    if base_cfg.sample is None:
        raise ValueError("Study base config must include a sample section.")
    expanded_trials = _expand_trials(spec)
    validate_study_factor_alignment(cfg=base_cfg, spec=spec, trials=expanded_trials)
    target_sets = _resolve_target_sets(spec, base_cfg)
    _preflight_trial_config_contracts(spec, base_cfg, trials=expanded_trials, target_sets=target_sets)
    target_cfg = _config_with_regulator_sets(base_cfg, target_sets)
    _ensure_lock_parse_and_targets_ready(target_cfg, base_config_path)
    base_config_sha = sha256_path(base_config_path)
    spec_sha = _spec_hash(spec)
    study_id = _compute_study_id(spec, base_config_sha256=base_config_sha)
    study_run_dir = resolve_study_run_dir(workspace_root, spec.name, study_id)
    return _StudyBootstrap(
        resolved_spec_path=resolved_spec_path,
        spec=spec,
        base_config_path=base_config_path,
        workspace_root=workspace_root,
        base_cfg=base_cfg,
        expanded_trials=expanded_trials,
        target_sets=target_sets,
        base_config_sha=base_config_sha,
        spec_sha=spec_sha,
        study_id=study_id,
        study_run_dir=study_run_dir,
        manifest_file=study_manifest_path(study_run_dir),
        status_file=study_status_path(study_run_dir),
    )


def _prepare_study_run_dir(*, study_run_dir: Path, resume: bool, force_overwrite: bool) -> None:
    if study_run_dir.exists():
        if force_overwrite:
            shutil.rmtree(study_run_dir)
            return
        if not resume:
            raise ValueError(f"Study run directory already exists: {study_run_dir}. Use --resume or --force-overwrite.")
        return
    if resume:
        raise FileNotFoundError(f"Cannot resume missing study run directory: {study_run_dir}")


def _initialize_study_state(
    *,
    bootstrap: _StudyBootstrap,
    resume: bool,
) -> tuple[StudyManifestV1, StudyStatusV1]:
    if resume:
        manifest = load_study_manifest(bootstrap.manifest_file)
        status = load_study_status(bootstrap.status_file)
        if manifest.spec_sha256 != bootstrap.spec_sha:
            raise ValueError("Cannot resume: spec hash does not match existing study manifest.")
        expected_runs = _resolve_trial_runs(
            trials=bootstrap.expanded_trials,
            seeds=bootstrap.spec.replicates.seeds,
            target_sets=bootstrap.target_sets,
        )
        manifest.trial_runs = _merge_resume_state(expected_runs, manifest.trial_runs)
        return manifest, status

    bootstrap.study_run_dir.mkdir(parents=True, exist_ok=True)
    study_tables_dir(bootstrap.study_run_dir).mkdir(parents=True, exist_ok=True)
    study_plots_dir(bootstrap.study_run_dir).mkdir(parents=True, exist_ok=True)
    payload = _spec_payload(bootstrap.spec)
    atomic_write_yaml(spec_frozen_path(bootstrap.study_run_dir), payload, sort_keys=False, default_flow_style=False)
    trial_runs = _resolve_trial_runs(
        trials=bootstrap.expanded_trials,
        seeds=bootstrap.spec.replicates.seeds,
        target_sets=bootstrap.target_sets,
    )
    manifest = StudyManifestV1(
        study_name=bootstrap.spec.name,
        study_id=bootstrap.study_id,
        spec_path=str(bootstrap.resolved_spec_path),
        spec_sha256=bootstrap.spec_sha,
        base_config_path=str(bootstrap.base_config_path),
        base_config_sha256=bootstrap.base_config_sha,
        created_at=utc_now_iso(),
        trial_runs=trial_runs,
    )
    status = StudyStatusV1(
        study_name=bootstrap.spec.name,
        study_id=bootstrap.study_id,
        status="running",
        started_at=utc_now_iso(),
        updated_at=utc_now_iso(),
    )
    return manifest, status


def _persist_study_state(
    *,
    manifest_file: Path,
    status_file: Path,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
) -> None:
    write_study_manifest(manifest_file, manifest)
    write_study_status(status_file, status)


def _run_study_trials(
    *,
    bootstrap: _StudyBootstrap,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
    progress_bar: bool,
    quiet_logs: bool,
) -> tuple[bool, bool]:
    if int(bootstrap.spec.execution.parallelism) <= 1:
        return _run_study_trials_serial(
            bootstrap=bootstrap,
            manifest=manifest,
            status=status,
            progress_bar=progress_bar,
            quiet_logs=quiet_logs,
        )
    return _run_study_trials_parallel(
        bootstrap=bootstrap,
        manifest=manifest,
        status=status,
    )


def _trial_run_is_complete(trial_run: StudyTrialRun) -> bool:
    return trial_run.status == "success" and bool(trial_run.run_dir)


def _log_parallel_trial_progress(
    *,
    status: StudyStatusV1,
    total_runs: int,
    queued_count: int,
) -> None:
    completed = int(status.success_runs + status.error_runs + status.skipped_runs)
    logger.info(
        "Study trial progress: completed=%d/%d success=%d error=%d running=%d queued=%d.",
        completed,
        int(total_runs),
        int(status.success_runs),
        int(status.error_runs),
        int(status.running_runs),
        int(queued_count),
    )


def _shutdown_trial_worker_process(process: multiprocessing.Process, *, terminate: bool) -> None:
    if terminate and process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join()
        return
    process.join()


def _run_study_trials_serial(
    *,
    bootstrap: _StudyBootstrap,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
    progress_bar: bool,
    quiet_logs: bool,
) -> tuple[bool, bool]:
    any_errors = False
    aborted = False
    for idx, trial_run in enumerate(manifest.trial_runs):
        if _trial_run_is_complete(trial_run):
            continue
        trial_run.status = "running"
        trial_run.error = None
        trial_run.started_at = utc_now_iso()
        trial_run.finished_at = None
        manifest.trial_runs[idx] = trial_run
        _refresh_status(status, manifest)
        _persist_study_state(
            manifest_file=bootstrap.manifest_file,
            status_file=bootstrap.status_file,
            manifest=manifest,
            status=status,
        )

        try:
            _append_log(
                bootstrap.study_run_dir,
                f"RUN trial={trial_run.trial_id} seed={trial_run.seed} target_set={trial_run.target_set_index}",
            )
            run_dir = _run_one_trial(
                base_cfg=bootstrap.base_cfg,
                base_config_path=bootstrap.base_config_path,
                workspace_root=bootstrap.workspace_root,
                spec=bootstrap.spec,
                study_run_dir=bootstrap.study_run_dir,
                trial_run=trial_run,
                progress_bar=progress_bar,
                quiet_logs=quiet_logs,
            )
            trial_run.run_dir = str(run_dir)
            trial_run.status = "success"
        except Exception as exc:
            trial_run.status = "error"
            trial_run.error = str(exc)
            any_errors = True
            _append_log(
                bootstrap.study_run_dir,
                "ERROR "
                f"trial={trial_run.trial_id} seed={trial_run.seed} "
                f"target_set={trial_run.target_set_index}: {exc}",
            )
            if bootstrap.spec.execution.on_trial_error == "abort":
                aborted = True
        finally:
            trial_run.finished_at = utc_now_iso()
            manifest.trial_runs[idx] = trial_run
            _refresh_status(status, manifest)
            _persist_study_state(
                manifest_file=bootstrap.manifest_file,
                status_file=bootstrap.status_file,
                manifest=manifest,
                status=status,
            )
        if aborted:
            break
    if aborted:
        _mark_pending_as_skipped(
            manifest,
            reason="Skipped because execution aborted after earlier trial error.",
        )
    return any_errors, aborted


def _run_study_trials_parallel(
    *,
    bootstrap: _StudyBootstrap,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
) -> tuple[bool, bool]:
    any_errors = False
    aborted = False
    pending_indexes = [
        idx for idx, trial_run in enumerate(manifest.trial_runs) if not _trial_run_is_complete(trial_run)
    ]
    if not pending_indexes:
        return any_errors, aborted

    worker_count = min(int(bootstrap.spec.execution.parallelism), len(pending_indexes))
    logger.info("Study trial phase running with parallelism=%d.", worker_count)
    _append_log(
        bootstrap.study_run_dir,
        f"RUN_PARALLEL_START workers={worker_count} total={len(pending_indexes)}",
    )
    queued_indexes = list(pending_indexes)
    in_flight: dict[int, _TrialWorkerProcess] = {}
    ctx = multiprocessing.get_context("spawn")
    worker_module = importlib.import_module("dnadesign.cruncher.app.study_workflow")
    worker_target = getattr(worker_module, "_run_one_trial_worker_process_entry")
    last_progress_heartbeat = time.monotonic()
    force_terminate = False

    try:
        while queued_indexes or in_flight:
            while not aborted and queued_indexes and len(in_flight) < worker_count:
                idx = queued_indexes.pop(0)
                trial_run = manifest.trial_runs[idx]
                trial_run.status = "running"
                trial_run.error = None
                trial_run.started_at = utc_now_iso()
                trial_run.finished_at = None
                manifest.trial_runs[idx] = trial_run
                _refresh_status(status, manifest)
                _persist_study_state(
                    manifest_file=bootstrap.manifest_file,
                    status_file=bootstrap.status_file,
                    manifest=manifest,
                    status=status,
                )
                _append_log(
                    bootstrap.study_run_dir,
                    f"RUN trial={trial_run.trial_id} seed={trial_run.seed} target_set={trial_run.target_set_index}",
                )
                result_path = _trial_worker_result_path(study_run_dir=bootstrap.study_run_dir, trial_run=trial_run)
                if result_path.exists():
                    result_path.unlink()
                process = ctx.Process(
                    target=worker_target,
                    kwargs={
                        "base_config_path": str(bootstrap.base_config_path),
                        "workspace_root": str(bootstrap.workspace_root),
                        "study_run_dir": str(bootstrap.study_run_dir),
                        "trial_run_payload": trial_run.model_dump(mode="python"),
                        "seed_path": str(bootstrap.spec.replicates.seed_path),
                        "profile": str(bootstrap.spec.artifacts.trial_output_profile),
                        "progress_bar": False,
                        "quiet_logs": True,
                        "result_path": str(result_path),
                    },
                )
                process.start()
                in_flight[idx] = _TrialWorkerProcess(
                    process=process,
                    result_path=result_path,
                    trial_index=idx,
                )

            if not in_flight:
                break

            completed_indexes = [idx for idx, state in in_flight.items() if not state.process.is_alive()]
            if not completed_indexes:
                now = time.monotonic()
                if now - last_progress_heartbeat >= 30.0:
                    _log_parallel_trial_progress(
                        status=status,
                        total_runs=len(manifest.trial_runs),
                        queued_count=len(queued_indexes),
                    )
                    last_progress_heartbeat = now
                time.sleep(0.05)
                continue

            for idx in completed_indexes:
                state = in_flight.pop(idx)
                _shutdown_trial_worker_process(state.process, terminate=False)
                trial_run = manifest.trial_runs[idx]
                try:
                    ok, payload = _load_worker_result(state.result_path, exitcode=state.process.exitcode)
                    if ok:
                        trial_run.run_dir = payload
                        trial_run.status = "success"
                        _append_log(
                            bootstrap.study_run_dir,
                            "RUN_DONE "
                            f"trial={trial_run.trial_id} seed={trial_run.seed} target_set={trial_run.target_set_index}",
                        )
                    else:
                        trial_run.status = "error"
                        trial_run.error = payload
                        any_errors = True
                        _append_log(
                            bootstrap.study_run_dir,
                            "ERROR "
                            f"trial={trial_run.trial_id} seed={trial_run.seed} "
                            f"target_set={trial_run.target_set_index}: {payload}",
                        )
                        if bootstrap.spec.execution.on_trial_error == "abort":
                            aborted = True
                except Exception as exc:
                    trial_run.status = "error"
                    trial_run.error = str(exc)
                    any_errors = True
                    _append_log(
                        bootstrap.study_run_dir,
                        "ERROR "
                        f"trial={trial_run.trial_id} seed={trial_run.seed} "
                        f"target_set={trial_run.target_set_index}: {exc}",
                    )
                    if bootstrap.spec.execution.on_trial_error == "abort":
                        aborted = True
                finally:
                    trial_run.finished_at = utc_now_iso()
                    manifest.trial_runs[idx] = trial_run
                    _refresh_status(status, manifest)
                    _persist_study_state(
                        manifest_file=bootstrap.manifest_file,
                        status_file=bootstrap.status_file,
                        manifest=manifest,
                        status=status,
                    )
            _log_parallel_trial_progress(
                status=status,
                total_runs=len(manifest.trial_runs),
                queued_count=len(queued_indexes),
            )
            last_progress_heartbeat = time.monotonic()
    except Exception:
        force_terminate = True
        raise
    finally:
        terminate_running = bool(force_terminate or aborted)
        for idx, state in sorted(in_flight.items(), key=lambda item: item[0]):
            _shutdown_trial_worker_process(state.process, terminate=terminate_running)
            if not terminate_running:
                continue
            trial_run = manifest.trial_runs[idx]
            if trial_run.status != "running":
                continue
            trial_run.status = "skipped"
            trial_run.error = "Skipped because execution aborted before trial completion."
            trial_run.finished_at = utc_now_iso()
            manifest.trial_runs[idx] = trial_run
        if terminate_running and in_flight:
            _refresh_status(status, manifest)
            _persist_study_state(
                manifest_file=bootstrap.manifest_file,
                status_file=bootstrap.status_file,
                manifest=manifest,
                status=status,
            )

    if aborted:
        _mark_pending_as_skipped(
            manifest,
            reason="Skipped because execution aborted after earlier trial error.",
        )
    return any_errors, aborted


def _run_study_replays(
    *,
    bootstrap: _StudyBootstrap,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
    aborted: bool,
) -> tuple[bool, bool]:
    any_errors = False
    if not bootstrap.spec.replays.mmr_sweep.enabled or aborted:
        return any_errors, aborted
    replay_candidates = [item for item in manifest.trial_runs if item.status == "success" and bool(item.run_dir)]
    total_replays = len(replay_candidates)
    logger.info("Study replay phase starting: %d successful trial run(s).", total_replays)
    _append_log(
        bootstrap.study_run_dir,
        f"REPLAY_START total={total_replays}",
    )
    completed = 0
    for idx, trial_run in enumerate(manifest.trial_runs):
        if trial_run.status != "success" or not trial_run.run_dir:
            continue
        run_dir = Path(trial_run.run_dir)
        logger.info(
            "Study replay run %d/%d: trial=%s seed=%d target_set=%d",
            completed + 1,
            total_replays,
            trial_run.trial_id,
            int(trial_run.seed),
            int(trial_run.target_set_index),
        )
        try:
            run_mmr_sweep_for_run(
                run_dir,
                pool_size_values=bootstrap.spec.replays.mmr_sweep.pool_size_values,
                diversity_values=bootstrap.spec.replays.mmr_sweep.diversity_values,
            )
            completed += 1
            _append_log(
                bootstrap.study_run_dir,
                f"REPLAY_DONE trial={trial_run.trial_id} seed={trial_run.seed} target_set={trial_run.target_set_index}",
            )
        except Exception as exc:
            any_errors = True
            trial_run.status = "error"
            trial_run.error = f"MMR replay failed: {exc}"
            trial_run.finished_at = utc_now_iso()
            manifest.trial_runs[idx] = trial_run
            _append_log(
                bootstrap.study_run_dir,
                "ERROR replay "
                f"trial={trial_run.trial_id} seed={trial_run.seed} "
                f"target_set={trial_run.target_set_index}: {exc}",
            )
            _refresh_status(status, manifest)
            _persist_study_state(
                manifest_file=bootstrap.manifest_file,
                status_file=bootstrap.status_file,
                manifest=manifest,
                status=status,
            )
            if bootstrap.spec.execution.on_trial_error == "abort":
                aborted = True
                _mark_pending_as_skipped(
                    manifest,
                    reason="Skipped because execution aborted after replay error.",
                )
                break
    logger.info("Study replay phase complete: %d/%d replay run(s) finished.", completed, total_replays)
    return any_errors, aborted


def _maybe_summarize_study(
    *,
    bootstrap: _StudyBootstrap,
    manifest: StudyManifestV1,
    status: StudyStatusV1,
) -> None:
    if not bootstrap.spec.execution.summarize_after_run:
        return
    has_non_success = any(item.status != "success" for item in manifest.trial_runs)
    if has_non_success:
        warning = (
            "Summary skipped due trial errors. "
            "Run `cruncher study summarize --allow-partial --run <study_run_dir>` to summarize successes."
        )
        if warning not in status.warnings:
            status.warnings.append(warning)
        status.updated_at = utc_now_iso()
        write_study_status(bootstrap.status_file, status)
        return
    logger.info("Study summarize phase starting.")
    _append_log(bootstrap.study_run_dir, "SUMMARIZE_START")
    summarize_study_run(bootstrap.study_run_dir, allow_partial=False)
    _append_log(bootstrap.study_run_dir, "SUMMARIZE_DONE")
    logger.info("Study summarize phase complete.")


def run_study(
    spec_path: Path,
    *,
    resume: bool = False,
    force_overwrite: bool = False,
    progress_bar: bool = True,
    quiet_logs: bool = False,
) -> Path:
    if resume and force_overwrite:
        raise ValueError("Use either resume or force_overwrite, not both.")
    bootstrap = _bootstrap_study_inputs(spec_path)
    _prepare_study_run_dir(study_run_dir=bootstrap.study_run_dir, resume=resume, force_overwrite=force_overwrite)
    manifest, status = _initialize_study_state(bootstrap=bootstrap, resume=resume)
    _refresh_status(status, manifest)
    _persist_study_state(
        manifest_file=bootstrap.manifest_file,
        status_file=bootstrap.status_file,
        manifest=manifest,
        status=status,
    )

    any_errors = False
    aborted = False
    fatal_exc: Exception | None = None
    try:
        trial_errors, aborted = _run_study_trials(
            bootstrap=bootstrap,
            manifest=manifest,
            status=status,
            progress_bar=progress_bar,
            quiet_logs=quiet_logs,
        )
        any_errors = any_errors or trial_errors
        replay_errors, aborted = _run_study_replays(
            bootstrap=bootstrap,
            manifest=manifest,
            status=status,
            aborted=aborted,
        )
        any_errors = any_errors or replay_errors
        _maybe_summarize_study(bootstrap=bootstrap, manifest=manifest, status=status)
    except Exception as exc:
        fatal_exc = exc
    finally:
        _refresh_status(status, manifest, final=True, failed=aborted or fatal_exc is not None)
        _persist_study_state(
            manifest_file=bootstrap.manifest_file,
            status_file=bootstrap.status_file,
            manifest=manifest,
            status=status,
        )

    if fatal_exc is not None:
        raise fatal_exc
    if any_errors and bootstrap.spec.execution.exit_code_policy == "nonzero_if_any_error":
        raise RuntimeError(f"Study completed with trial errors. See {bootstrap.status_file} for details.")
    return bootstrap.study_run_dir


def study_show_payload(study_run_dir: Path) -> dict[str, object]:
    manifest = load_study_manifest(study_manifest_path(study_run_dir))
    status = load_study_status(study_status_path(study_run_dir))
    plots = sorted(study_plots_dir(study_run_dir).glob(study_plot_glob(study_run_dir)))
    tables = sorted(study_tables_dir(study_run_dir).glob("table__*"))
    return {
        "study_name": manifest.study_name,
        "study_id": manifest.study_id,
        "status": status.status,
        "total_runs": status.total_runs,
        "success_runs": status.success_runs,
        "error_runs": status.error_runs,
        "pending_runs": status.pending_runs,
        "plot_paths": [str(path) for path in plots],
        "table_paths": [str(path) for path in tables],
        "manifest_path": str(study_manifest_path(study_run_dir)),
        "status_path": str(study_status_path(study_run_dir)),
    }
