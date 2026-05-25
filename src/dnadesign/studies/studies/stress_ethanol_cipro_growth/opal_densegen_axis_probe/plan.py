"""Study-owned DenseGen axis OPAL probe package."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .artifacts import ProbeArtifactLayout, ProbePlan, RunSpec
from .constants import CAMPAIGNS, DEFAULT_TOP_K, ORACLE_ID, ORACLES, RUN_STAGES, SHARED_OBSERVED_LABEL_SIDECAR, SPLITS
from .suite_manifest import default_probe_suite


def _validate_rounds(rounds: int) -> int:
    value = int(rounds)
    if value < 1:
        raise ValueError("--rounds must be >= 1")
    return value


def _gate_matrix(gate: str | None, splits: tuple[str, ...]) -> list[tuple[str, str, str]]:
    gate = (gate or "all").strip().lower()
    if gate == "source":
        return []
    if gate == "cipro-random":
        return [("cipro", oracle, "random_id") for oracle in ORACLES]
    if gate == "random-all":
        return [(campaign, oracle, "random_id") for campaign in CAMPAIGNS for oracle in ORACLES]
    if gate == "leave-sigma35":
        return [(campaign, oracle, "leave_sigma35_variant") for campaign in CAMPAIGNS for oracle in ORACLES]
    if gate != "all":
        raise ValueError(f"unknown gate: {gate}")
    requested_splits = tuple(dict.fromkeys(splits))
    return [(campaign, oracle, split) for campaign in CAMPAIGNS for oracle in ORACLES for split in requested_splits]


def _run_key(campaign_key: str, oracle_id: str, split_id: str) -> str:
    oracle_short = "positive" if oracle_id == ORACLE_ID else "null"
    return f"{campaign_key}_{oracle_short}_{split_id}"


def _validate_stop_after(stop_after: str) -> str:
    value = str(stop_after or "status").strip().lower()
    if value not in RUN_STAGES:
        raise ValueError(f"unsupported stop_after={stop_after!r}; expected one of {', '.join(RUN_STAGES)}")
    return value


def _label_input_path(config_path: Path, round_index: int) -> Path:
    return config_path.parent.parent / "inputs" / f"r{int(round_index)}" / f"vec8-b{int(round_index)}.parquet"


def _opal_validate_command(config_path: Path) -> list[str]:
    config = str(config_path)
    return ["uv", "run", "opal", "validate", "-c", config]


def _opal_init_command(config_path: Path) -> list[str]:
    config = str(config_path)
    return ["uv", "run", "opal", "init", "-c", config]


def _opal_ingest_command(config_path: Path, round_index: int) -> list[str]:
    config = str(config_path)
    return [
        "uv",
        "run",
        "opal",
        "ingest-y",
        "-c",
        config,
        "--round",
        str(int(round_index)),
        "--in",
        str(_label_input_path(config_path, int(round_index))),
        "--unknown-sequences",
        "error",
        "--apply",
    ]


def _opal_run_command(config_path: Path, round_index: int) -> list[str]:
    config = str(config_path)
    return ["uv", "run", "opal", "run", "-c", config, "--round", str(int(round_index)), "--resume", "--json"]


def _opal_status_command(config_path: Path) -> list[str]:
    config = str(config_path)
    return ["uv", "run", "opal", "status", "-c", config, "--with-ledger", "--json"]


def _opal_commands(config_path: Path, *, stop_after: str = "status", rounds: int = 1) -> list[list[str]]:
    stop = _validate_stop_after(stop_after)
    round_count = _validate_rounds(rounds)
    commands: list[list[str]] = []
    if stop == "materialize":
        return commands
    commands.append(_opal_validate_command(config_path))
    if RUN_STAGES.index(stop) < RUN_STAGES.index("init"):
        return commands
    commands.append(_opal_init_command(config_path))
    if RUN_STAGES.index(stop) < RUN_STAGES.index("ingest"):
        return commands
    if RUN_STAGES.index(stop) < RUN_STAGES.index("run"):
        commands.append(_opal_ingest_command(config_path, 0))
        return commands
    for round_index in range(round_count):
        commands.append(_opal_ingest_command(config_path, round_index))
        commands.append(_opal_run_command(config_path, round_index))
    if RUN_STAGES.index(stop) >= RUN_STAGES.index("status"):
        commands.append(_opal_status_command(config_path))
    return commands


def build_plan(
    *,
    run_root: Path,
    initial_label_count: int,
    seed: int,
    gate: str | None,
    splits: Iterable[str],
    selection_k: int = DEFAULT_TOP_K,
    max_x_matrix_gib: float | None = None,
    score_batch_size: int | None = None,
    rounds: int = 1,
    apply: bool = False,
    stop_after: str = "status",
    suite_id: str | None = None,
) -> ProbePlan:
    suite = default_probe_suite()
    stop = _validate_stop_after(stop_after)
    round_count = _validate_rounds(rounds)
    initial_count = int(initial_label_count)
    if initial_count < 1:
        raise ValueError("--initial-labels must be >= 1")
    batch_size = int(selection_k)
    if batch_size < 1:
        raise ValueError("--selection-k must be >= 1")
    memory_budget = None if max_x_matrix_gib is None else float(max_x_matrix_gib)
    if memory_budget is not None and memory_budget <= 0.0:
        raise ValueError("--max-x-matrix-gib must be > 0")
    score_batch = None if score_batch_size is None else int(score_batch_size)
    if score_batch is not None and score_batch < 1:
        raise ValueError("--score-batch-size must be >= 1")
    split_tuple = tuple(dict.fromkeys(str(split).strip() for split in splits if str(split).strip()))
    if not split_tuple:
        split_tuple = ("random_id", "leave_sigma35_variant")
    invalid_splits = sorted(set(split_tuple) - set(SPLITS))
    if invalid_splits:
        raise ValueError(f"unsupported split(s): {invalid_splits}")

    layout = ProbeArtifactLayout(run_root)
    runs: list[RunSpec] = []
    commands: list[list[str]] = []
    for campaign_key, oracle_id, split_id in _gate_matrix(gate, split_tuple):
        run_key = _run_key(campaign_key, oracle_id, split_id)
        workdir = layout.campaign_workdir(run_key)
        config_path = layout.campaign_config_path(run_key)
        label_input_path = layout.campaign_label_input_path(run_key)
        sidecar_path = layout.campaign_sidecar_path(run_key, split_id)
        runs.append(
            RunSpec(
                campaign_key=campaign_key,
                oracle_id=oracle_id,
                split_id=split_id,
                run_key=run_key,
                target_class=str(CAMPAIGNS[campaign_key]["target_class"]),
                workdir=workdir,
                config_path=config_path,
                label_input_path=label_input_path,
                sidecar_path=sidecar_path,
                selection_k=batch_size,
                seed=int(seed),
                label_family_id=suite.active_label_family,
                max_x_matrix_gib=memory_budget,
                score_batch_size=score_batch,
            )
        )
        commands.extend(_opal_commands(config_path, stop_after=stop, rounds=round_count))
    return ProbePlan(
        run_root=run_root,
        initial_label_count=initial_count,
        selection_k=batch_size,
        max_x_matrix_gib=memory_budget,
        score_batch_size=score_batch,
        seed=int(seed),
        rounds=round_count,
        gate=gate,
        splits=split_tuple,
        apply=bool(apply),
        stop_after=stop,
        suite_id=str(suite_id or suite.suite_id),
        suite_seeds=tuple(int(value) for value in suite.seeds),
        active_label_family=suite.active_label_family,
        passive_label_families=suite.passive_label_families,
        runs=runs,
        commands=commands,
    )


def validate_scratch_paths(*, run_root: Path, label_sidecar_path: Path) -> None:
    run_root_resolved = run_root.resolve()
    sidecar_resolved = label_sidecar_path.resolve()
    shared_suffix = Path(SHARED_OBSERVED_LABEL_SIDECAR)
    if str(sidecar_resolved).endswith(str(shared_suffix)):
        raise ValueError(f"refusing shared observed-label sidecar path: {sidecar_resolved}")
    try:
        sidecar_resolved.relative_to(run_root_resolved)
    except ValueError as exc:
        raise ValueError(f"scratch label sidecar is outside run root: {sidecar_resolved}") from exc
