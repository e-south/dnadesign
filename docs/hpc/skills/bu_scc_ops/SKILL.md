---
name: BU SCC Ops
description: This skill should be used when the user asks to "run on BU SCC", "start an interactive SCC session", "submit a qsub job", "pick SCC resources", "set DenseGen CPU resources", "run Evo2 on SCC GPUs", or "set up Notify watcher jobs on SCC".
version: 0.1.0
---

# BU SCC Ops Skill

Use this skill to run `dnadesign` workloads on BU SCC with explicit scheduler resources, predictable job behavior, and clear Notify wiring.

## Scope

Use this skill for:
- interactive SCC sessions (`qrsh`, OnDemand shell)
- batch job submission (`qsub`) for DenseGen, Evo2, and Notify watcher flows
- choosing resource requests by workload type
- preflight validation commands before long runs

Do not use this skill for:
- non-SCC environments
- generic local development workflow decisions

## Workflow

1. Identify workload class first.
2. Pick interactive or batch mode.
3. Apply resource profile from `references/resource_profiles.md`.
4. Use template command from `docs/hpc/agent_cheatsheet.md`.
5. Run preflight checks before expensive jobs.
6. Submit and monitor with `qstat` + log tail.

## Workload classes

- DenseGen solver runs:
  - CPU only
  - request `-pe omp <N>` and `-l mem_per_core=<...>`
  - ensure `densegen.solver.threads <= N`
- Notify watcher:
  - lightweight CPU watcher
  - long walltime, minimal slots
- Evo2 inference:
  - GPU required (`-l gpus=1 -l gpu_c=8.9`)
  - load CUDA/GCC modules in script
- large transfer/prefetch:
  - transfer-node queue (`-l download`)
  - no compute-heavy workloads

## Required checks

Before submitting DenseGen jobs:
- `uv run dense validate-config --probe-solver -c <config.yaml>`
- `uv run dense inspect config --probe-solver -c <config.yaml>`

Before relying on Notify:
- confirm events source is USR `.events.log`
- do not wire Notify to DenseGen `outputs/meta/events.jsonl`

## Command references

Load only the needed references:
- `docs/hpc/agent_cheatsheet.md` for copy/paste commands
- `docs/hpc/bu_scc_quickstart.md` for bootstrap flow
- `docs/hpc/bu_scc_batch_notify.md` for runbook details
- `docs/hpc/jobs/README.md` for template-specific submits
- `references/resource_profiles.md` for workload-to-resource defaults
