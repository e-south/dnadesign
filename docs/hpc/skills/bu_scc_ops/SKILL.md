---
name: BU SCC Ops
description: This skill should be used when the user asks to "run on BU SCC", "start an interactive SCC session", "submit a qsub job", "pick SCC resources", "set DenseGen CPU resources", "run Evo2 on SCC GPUs", or "set up Notify watcher jobs on SCC".
version: 0.1.0
---

# BU SCC Ops Skill

Use this skill to run `dnadesign` workloads on BU SCC with explicit scheduler resources, predictable job behavior, and clear Notify wiring.

Operational source of truth:
- `docs/hpc/bu_scc_ops_cheatsheet.md`

Keep this skill focused on triggering and execution flow. Do not duplicate command/resource defaults here.

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
3. Apply the matching row from `docs/hpc/bu_scc_ops_cheatsheet.md` task-to-resource mapping.
4. Use template command from `docs/hpc/bu_scc_ops_cheatsheet.md`.
5. Run preflight checks before expensive jobs.
6. Submit and monitor with `qstat` + log tail.
7. Keep interactive/OnDemand usage within BU policy limits for high-resource sessions.

## Workload classes

- DenseGen solver runs: CPU-only, thread count must align with requested `omp` slots.
- Notify watcher: low-footprint CPU watcher for USR event delivery.
- Evo2 inference: GPU-required workloads only.
- large transfer/prefetch: transfer-node queue only.

## Required checks

Before submitting DenseGen jobs:
- `uv run dense validate-config --probe-solver -c <config.yaml>`
- `uv run dense inspect config --probe-solver -c <config.yaml>`

Before relying on Notify:
- confirm events source is USR `.events.log`
- do not wire Notify to DenseGen `outputs/meta/events.jsonl`
- run resolver preflight: `uv run notify setup resolve-events --tool <tool> --config <config.yaml>`
- ensure events are schema-valid (must include `event_version`)

## General usage patterns

Use these patterns for common requests. Keep placeholders explicit and fill values before execution.

Interactive CPU debug session (1 hour):

```bash
qrsh -P <project> -l h_rt=01:00:00 -pe omp 8 -l mem_per_core=8G -cwd -now n
```

DenseGen batch submit (CPU):

```bash
qsub -P <project> \
  -pe omp 16 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<config.yaml> \
  docs/hpc/jobs/bu_scc_densegen_cpu.qsub
```

Notify preflight plus watcher setup:

```bash
uv run notify setup resolve-events --tool densegen --config <config.yaml>
uv run notify setup slack --tool densegen --config <config.yaml> --secret-source auto --policy densegen
```

Evo2 GPU submit:

```bash
qsub -P <project> \
  -v CUDA_MODULE=cuda/<version>,GCC_MODULE=gcc/<version> \
  docs/hpc/jobs/bu_scc_evo2_gpu_infer.qsub
```

Queue state checks:

```bash
qstat -u "$USER"
qstat -j <job_id>
```

For exact curated command sets, use `docs/hpc/bu_scc_ops_cheatsheet.md`.

## Command references

Load only the needed references:
- `docs/hpc/bu_scc_ops_cheatsheet.md` for copy/paste commands
- `docs/hpc/bu_scc_quickstart.md` for bootstrap flow
- `docs/hpc/bu_scc_batch_notify.md` for runbook details
- `docs/hpc/jobs/README.md` for template-specific submits

## Source links

- BU SCC OnDemand overview: https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/
- BU SCC My Interactive Sessions: https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/sessions/
- BU SCC interactive jobs: https://www.bu.edu/tech/support/research/system-usage/running-jobs/interactive-jobs/
- BU SCC submitting jobs: https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
- BU SCC technical summary: https://www.bu.edu/tech/support/research/system-usage/running-jobs/technical-summary/
