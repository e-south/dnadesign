---
name: sge-hpc-ops
description: >
  End-to-end operations guide for running workloads on BU SCC (SGE): boot interactive
  or batch sessions, request appropriate resources, submit jobs, monitor progress,
  and (when possible) attach or inspect running nodes. Includes dnadesign examples
  (DenseGen/Evo2/Notify) but stays workload-agnostic so the agent can generalize.
---

# SGE HPC Ops Skill

Operate BU SCC (SGE) workloads with predictable resources, clear job boundaries, and agent-friendly step-in or step-out monitoring.

This skill is designed for an agent that can run shell commands and keep subprocesses alive for watchers (queue polling, log tailing), so the user does not have to manually watch another terminal.

## Design goals and critical evaluation of the prior version

### What was good

- Strong workload class to resource request mapping.
- Clear core rules (explicit `-P`, explicit `h_rt`, thread alignment).
- Correct separation between DenseGen runtime diagnostics vs USR `.events.log` for Notify.

### What needed improvement (and is addressed below)

- Agent-operational gap: prior text said this session does not automatically jump into the new compute-node shell. This revision adds a concrete session-handle model and subprocess patterns.
- Too BU or dnadesign specific in core logic: mapping is kept, but rephrased as generic workload patterns with dnadesign examples.
- Monitoring guidance was human-terminal oriented: upgraded to an agent-run watch loop plus attach strategy (job id to exec host to addressable logs plus optional node inspection).
- Ambiguous or incomplete copy and paste blocks: fixed formatting and added terse job id patterns plus log discovery.
- Missing failure-mode playbook: adds Eqw handling, log-not-found handling, resource mismatch symptoms, and safe cancellation.

## Scope

Use this skill when the user asks to:

- run on BU SCC or SCC OnDemand
- start an interactive session (`qrsh`, OnDemand shell)
- submit or modify a batch job (`qsub`, arrays)
- choose SCC resources (CPU vs GPU vs low-footprint watchers vs transfer)
- wire watchers (Notify-style event tails) and report progress

Do not use this skill for:

- non-HPC or local-only workflows (unless asked to translate HPC concepts)
- decisions unrelated to scheduler sessions (for example, algorithm selection)

## Mental model (portable HPC language)

Even though BU SCC uses SGE, keep the language transferable:

- Login node: where you connect via SSH or OnDemand; do not do heavy compute here.
- Compute node: where jobs run after the scheduler starts them.
- Interactive job: you get a shell on a compute node (BU: `qrsh` or OnDemand interactive shell).
- Batch job: you submit a script; scheduler runs it; you watch logs (BU: `qsub`).
- Resources: walltime, CPUs or threads, memory, GPUs, special queues (transfer or download).

## Agent-first operating model

When the agent can execute commands, it should behave like an operator with state and handles, not like a copy or paste chatbot.

### Session handle (the addressable location contract)

For any SCC action, maintain a structured handle in working memory and summarize it to the user:

- `login_host`: for example `scc1.bu.edu`
- `project`: the `-P <project>` value
- `mode`: `interactive` or `batch`
- `job_id`: SGE job id once known
- `job_name`: if set
- `workdir`: where the job runs (`-cwd` or `#$ -cwd`)
- `log_path`: the primary stdout or stderr file path
- `node`: exec host when running (if discoverable)
- `watchers`: list of subprocess watchers started (queue poll, `tail -F`, and similar)

This enables step in and step out:

- Step in: run commands against the handle (`qstat`, read logs, optional ssh to node).
- Step out: stop watchers or summarize and return status with next actions.

### Subprocess watchers (preferred)

When possible, start at least one watcher subprocess:

- queue watcher: periodic `qstat` snapshots
- log watcher: `tail -n +1 -F <log_path>` once it exists

If the environment cannot keep subprocesses alive, fall back to poll-on-demand and ask the user to paste outputs.

## Workflow (operator loop)

1. Identify workload pattern:
   CPU compute (single-node, multi-thread), GPU compute, low-footprint watcher or daemon, transfer or download-only.
2. Choose mode:
   interactive for debug or smoke tests; batch for real runs, restart safety, or long walltimes.
3. Choose resources from mapping below.
4. Preflight checks (fast fail).
5. Start session or submit job.
6. Start watchers (agent subprocess when possible).
7. When running, capture node and log path and provide periodic summaries.
8. When finished, report exit status cues and where outputs live.

## Core rules (SGE and SCC safe)

- Always provide `-P <project>` (or `#$ -P <project>` in scripts).
- Always request walltime explicitly: `-l h_rt=HH:MM:SS`.
- Prefer shorter walltimes when feasible to improve scheduling.
- Always run jobs in a known directory:
  interactive: `qrsh ... -cwd`; batch scripts: include `#$ -cwd` (or pass `-cwd` at submit).
- Align threads with requested slots:
  keep app threads less than or equal to `-pe omp <slots>` and use `NSLOTS` in scripts when possible.
- Memory requests must be consistent with parallel slots:
  if using `mem_per_core`, total RAM scales with slots.
- Do not do compute-heavy runs on login or transfer nodes.
- For GPU workloads, request GPUs explicitly and ensure CUDA and toolchain modules match runtime needs.
- For watcher workloads, keep them small (1 slot, low memory) and long walltime if needed.

## Workload patterns and starter resource requests

These are starter values; scale based on observed memory, CPU, and GPU utilization.

| Pattern | Mode | Starter request (BU SCC / SGE) | Notes |
| --- | --- | --- | --- |
| CPU debug or smoke test | interactive | `-l h_rt=01:00:00 -pe omp 4 -l mem_per_core=4G` | Fast iteration, avoid huge requests. |
| CPU production (multi-thread) | batch | `-l h_rt=08:00:00 -pe omp 16 -l mem_per_core=8G` | Use `NSLOTS` and cap threads. |
| Low-footprint watcher or tailer | batch | `-l h_rt=24:00:00 -pe omp 1 -l mem_per_core=2G` | Good for Notify or watch loops. |
| GPU inference or training | batch (or interactive if allowed) | `-l h_rt=04:00:00 -pe omp 4 -l mem_per_core=8G -l gpus=1 -l gpu_c=8.9` | Adjust `gpu_c` and count to workload needs. |
| Download or transfer-only | batch (download queue) | `-l download -l h_rt=24:00:00 -pe omp 1` | Do not run compute-heavy tasks here. |

dnadesign examples:

- DenseGen: CPU production pattern
- Evo2 inference: GPU pattern
- Notify watcher: low-footprint watcher pattern
- Model or dataset prefetch: transfer or download-only pattern

## Required preflight checks (generic plus dnadesign examples)

### Generic fast-fail checks

Run on login node before submitting long jobs:

```bash
whoami
hostname
pwd
which qsub qrsh qstat || true
qstat -u "$USER" | head -n 20 || true
```

Check storage locations exist and are writable:

```bash
mkdir -p outputs/logs
touch outputs/logs/.write_test && rm outputs/logs/.write_test
```

### dnadesign-specific checks

Before long DenseGen runs:

```bash
uv run dense validate-config --probe-solver -c <config.yaml>
uv run dense inspect config --probe-solver -c <config.yaml>
```

Before relying on Notify watchers:

- Confirm events source is USR `.../.events.log` (JSONL).
- Do not wire Notify to DenseGen `outputs/meta/events.jsonl`.
- Resolve and validate wiring:

```bash
uv run notify setup resolve-events --tool <tool> --config <config.yaml>
```

## Interactive sessions (qrsh or OnDemand) with agent step-in/out

### Preferred for long interactive sessions: SCC OnDemand

- Use OnDemand interactive shell sessions for persistence.
- Agent role: guide selection, monitor job state, and keep notes of session URL or name.

```text
BU SCC OnDemand overview:
https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/

My Interactive Sessions:
https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/sessions/
```

### `qrsh` interactive (best for short debug)

```bash
qrsh -P <project> -l h_rt=01:00:00 -pe omp 8 -l mem_per_core=8G -cwd -now n
```

Agent subprocess guidance:

- Start `qrsh` as a dedicated subprocess or terminal session the agent controls.
- Once inside the compute node, record `hostname`, `pwd`, and key env vars (`NSLOTS`, `TMPDIR`).
- Run user commands inside that subprocess.
- Step out by leaving subprocess running while returning to chat with progress summaries.

Optional stability trick (not guaranteed):

```bash
tmux new -s work
# run commands...
```

## Batch submission (qsub) with agent-run monitoring

### Minimum robust SGE script header

```bash
#!/bin/bash -l
#$ -P <project>
#$ -N <job_name>
#$ -cwd
#$ -l h_rt=04:00:00
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -j y
#$ -o outputs/logs/$JOB_NAME.$JOB_ID.out

set -euo pipefail

echo "host: $(hostname)"
echo "workdir: $(pwd)"
echo "date: $(date)"
echo "NSLOTS=${NSLOTS:-unset}"
```

### Submit and capture job id

Use `-terse` when available to get a clean job id:

```bash
JOB_ID="$(qsub -terse -P <project> docs/hpc/jobs/<template>.qsub)"
echo "submitted job $JOB_ID"
```

If `-terse` is not supported, parse standard `qsub` output.

### dnadesign examples (illustrative)

DenseGen CPU batch submit:

```bash
qsub -P <project> \
  -pe omp 16 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<abs_path_to_config.yaml> \
  docs/hpc/jobs/bu_scc_densegen_cpu.qsub
```

Evo2 GPU submit:

```bash
qsub -P <project> \
  -v CUDA_MODULE=cuda/<version>,GCC_MODULE=gcc/<version> \
  docs/hpc/jobs/bu_scc_evo2_gpu_infer.qsub
```

Notify watcher submit:

```bash
qsub -P <project> \
  -v EVENTS_PATH=<abs_path_to_usr_events_log>,CURSOR_PATH=<abs_cursor>,SPOOL_DIR=<abs_spool>,WEBHOOK_ENV=<ENV_VAR_NAME> \
  docs/hpc/jobs/bu_scc_notify_watch.qsub
```

## Monitoring and control (agent-friendly)

### Job state quick reference (SGE)

- `qw`: queued or waiting
- `r`: running
- `Eqw`: error state (needs investigation)
- job disappears: finished or deleted (check accounting and logs)

### Core commands

```bash
qstat -u "$USER"
qstat -j <job_id>
```

### Agent watcher: queue poll loop

```bash
while true; do
  date
  qstat -u "$USER" | sed -n '1,20p'
  sleep 20
done
```

### Discover log path reliably

If scripts use a known `#$ -o ...` pattern, log path is deterministic.
If not, search common locations:

```bash
ls -lt outputs/logs | head
```

### Agent watcher: tail logs

```bash
tail -n +1 -F outputs/logs/<job_name>.<job_id>.out
```

### Investigate Eqw

```bash
qstat -j <job_id> | sed -n '1,200p'
# Look for error reason, failed resource requests, or path issues.
```

### Cancel safely

```bash
qdel <job_id>
```

## Step into the node strategies (realistic and policy-aware)

Attaching to a running batch job exact environment is not always supported. This skill supports three levels of step in:

1. Addressable log tail (always works).
   Treat log file as the primary place to step into. Tail and report.
2. Node identification plus inspection (sometimes works).
   When running, `qstat -j` may reveal execution host. If policy permits SSH to compute nodes from login nodes, inspect filesystem, processes, and GPU state as needed.
3. Interactive session as true step-in substrate (best UX).
   If user wants active command execution inside the run environment, prefer interactive jobs (`qrsh`, OnDemand). Treat batch as submit-and-observe.

Agent rule: if user asks to go into the node and run commands, prefer starting or recommending an interactive session unless cluster policy clearly supports attaching to batch.

Example pattern (only if allowed):

```bash
qstat -j <job_id> | grep -i -E "exec_host|hostname|queue"
# then (if permitted) ssh <exec_host>
```

## Transfer and download-node workflows

Use BU SCC download queue for large transfers:

```bash
qsub -l download <<'QSUB'
#!/bin/bash -l
#$ -P <project>
#$ -N transfer_job
#$ -cwd
#$ -l h_rt=24:00:00
#$ -pe omp 1
#$ -j y
#$ -o outputs/logs/transfer.$JOB_ID.out
set -euo pipefail

echo "host: $(hostname)"
echo "date: $(date)"

# transfer-only operations here (no compute-heavy work)
QSUB
```

## Anti-patterns and common mistakes (fast triage)

- Forgot `-cwd` or `#$ -cwd`: logs and outputs go to unexpected directories (often `$HOME`).
- Oversubscribed threads: app uses more threads than requested slots, causing contention and slowdowns.
- Walltime not set: defaults can be unfavorable and behavior less predictable.
- Running compute on login nodes: policy risk and unstable performance.
- Notify watching wrong file: must watch USR `.events.log` JSONL, not tool diagnostics logs.
- GPU job without correct module or toolchain: runtime import failures and CUDA or toolchain errors in logs.

## Reporting contract (what agent should always return)

After starting or submitting, report:

- mode: interactive or batch
- resources requested: walltime, slots, memory, GPUs
- job id (batch) or node hostname (interactive)
- log path (batch) or workdir (interactive)
- current state (`qw`, `r`, finished, `Eqw`)
- next action options (wait, adjust resources, debug via interactive, cancel)

Example status block:

```text
SCC status
- Mode: batch
- Job: 1234567 (name: densegen_run)
- State: qw (queued)
- Resources: h_rt=08:00:00, omp=16, mem_per_core=8G
- Log: outputs/logs/densegen_run.1234567.out (will appear when job starts)
- Watchers: qstat poll every 20s (active)
Next: if still queued after ~N minutes, consider lowering h_rt or slots, or switching to interactive smoke test.
```

## Command references (repo-local)

Prefer these docs and templates as canonical for dnadesign:

- `docs/hpc/bu_scc_quickstart.md` (bootstrap flow)
- `docs/hpc/bu_scc_install.md` (modules, uv, CUDA health)
- `docs/hpc/bu_scc_batch_notify.md` (batch and watcher runbook)
- `docs/hpc/jobs/README.md` (template-specific submits)

## BU SCC source references (operator policy and syntax)

```text
Submitting jobs:
https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/

Interactive jobs:
https://www.bu.edu/tech/support/research/system-usage/running-jobs/interactive-jobs/

Technical summary:
https://www.bu.edu/tech/support/research/system-usage/running-jobs/technical-summary/
```
