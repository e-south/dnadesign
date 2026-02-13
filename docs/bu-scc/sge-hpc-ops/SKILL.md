---
name: sge-hpc-ops
description: >
  Use this skill when the user asks about SGE/UGE/Grid Engine operations such as:
  "qsub", "qrsh", "qlogin", "qstat", "qdel", "job arrays (-t / SGE_TASK_ID)",
  "parallel environments (-pe)", "pick CPU/memory/GPU resources", "watch queue / tail logs",
  "debug Eqw", "transfer/download queues", or "run something on an SGE cluster".
  This skill is site-agnostic by default and BU-hosted in this repository.
  Keep scheduler-generic behavior in this file. For BU SCC platform specifics,
  load adjacent docs in ../ (README, quickstart, install, batch-notify, jobs/README).
  For dnadesign workload specifics, load references/workload-dnadesign.md.
---

# SGE HPC Ops

A **site-agnostic**, operator-grade playbook for **SGE-family schedulers** (Sun Grid Engine / Univa Grid Engine / compatible).
It is designed to be durable across clusters and future workloads.
In this repository, the skill is BU-hosted: keep generic scheduler behavior here and keep BU policy in adjacent `../` platform docs.

**Core principle:** **Probe capabilities first**, then generate commands using a *capability map*.
Do not emit cluster-specific flags (accounts, PE names, mem keys, GPU keys, special queues) until they are confirmed.

## What this skill is for

Use when the user wants to:
- start **interactive** compute sessions (e.g. `qrsh` / `qlogin`)
- submit **batch jobs** (`qsub`) and arrays
- choose resource requests (CPU, memory, GPU, walltime)
- monitor jobs (queue state, logs), diagnose failures (e.g. `Eqw`)
- run "watcher"/daemon-style jobs (lightweight tails, event watchers)
- handle transfer/download-node style jobs *when the site exposes them*

Not for:
- Slurm/PBS-only instructions (unless translating concepts at a high level)
- local-only development

## Non-stale design (how to keep this future-proof)

- **Keep core portable.** Keep platform policy and site keys in platform docs; keep workload-specific examples in `references/`.
- **Prefer placeholders + probing** over hard-coded flags.
- **Prefer discovery** (repo root, template locations) over fixed paths.
- **Bound watchers.** Never create unbounded loops without a stop/teardown rule.

## Operator objects to maintain

### 1) Capability map (mandatory)

A small mapping that records what *this* cluster supports and the correct resource key names.

Minimum fields (extend as needed):
- `accounting`: `{ supported: bool, flag: "-P" | "-A" | ..., notes }`
- `interactive_cmd`: `"qrsh"` or `"qlogin"` (or unsupported)
- `pe`: `{ supported: bool, flag: "-pe", names: [...], default_pe }`
- `walltime`: `{ supported: bool, key: "h_rt"|..., examples }`
- `memory`: `{ supported: bool, style: "per_core"|"total", key: "mem_per_core"|..., examples }`
- `gpu`: `{ supported: bool, keys: [...], examples }`
- `transfer`: `{ supported: bool, key_or_queue: "...", examples }`
- `policy`: `{ ssh_to_compute_allowed: "unknown"|"yes"|"no", notes }`

### 2) Run handle (per job/session)

- `mode`: `interactive` | `batch`
- `job_id` (batch) or `session_id` (interactive/OnDemand)
- `job_name`
- `workdir`
- `log_path` (stdout/stderr)
- `exec_host` (if discoverable)
- `watchers`: list of watcher subprocesses with **timeout + teardown plan**

## Step 0: Mandatory capability probe (before giving commands)

Run the probe on the login node (or equivalent control host).

### Probe A: confirm scheduler + available commands

```bash
command -v qsub qstat qdel >/dev/null && echo "SGE-like tools present"
command -v qrsh >/dev/null && echo "qrsh available" || true
command -v qlogin >/dev/null && echo "qlogin available" || true

qsub -help 2>&1 | sed -n '1,120p'
qstat -help 2>&1 | sed -n '1,120p'
```

### Probe-fail contract (mandatory)

If Probe A does not confirm `qsub` and `qstat`, stop command generation and ask the user for the correct scheduler context.
Do not emit SGE-specific flags or templates when scheduler identity is unknown.

### Probe B: discover parallel environments (PE) and queues (best effort)

Some sites allow `qconf`; some restrict it. Treat failures as "unknown".

```bash
qconf -spl 2>/dev/null || echo "[warn] qconf -spl not permitted; PE list unknown"
qconf -sql 2>/dev/null || echo "[warn] qconf -sql not permitted; queue list unknown"
```

### Probe C: discover resource keys (memory / gpu / special)

```bash
qconf -sc 2>/dev/null | sed -n '1,200p' || echo "[warn] qconf -sc not permitted; resource keys unknown"
```

Heuristic greps (safe; may return nothing):

```bash
qconf -sc 2>/dev/null | grep -i -E 'mem|vmem|h_vmem|mem_free|per_core' || true
qconf -sc 2>/dev/null | grep -i -E 'gpu|gpus|cuda' || true
qconf -sc 2>/dev/null | grep -i -E 'download|transfer|globus' || true
```

### Probe D: accounting/project flag detection (do not assume)

Look for project/account flags in `qsub -help`. If not obvious, treat as unknown and avoid specifying.

```bash
qsub -help 2>&1 | grep -E ' -P | -A |project|account' || true
```

**Rule:** if the accounting flag is not confirmed, do not include one in generated commands. Ask for site docs or a known "submit example" from the user.

## Step 1: Choose an execution mode (interactive vs batch)

### Choose interactive when:

- debugging environment issues
- iterating quickly
- you need a real shell "inside the node" as the primary workspace

### Choose batch when:

- runs are long
- you need restart safety + durable logs
- you want deterministic scheduler-managed resources
- you want to run watchers/daemons without keeping an interactive terminal open

## Step 2: Pick a workload pattern (portable categories)

Use these categories; do not tie the core logic to any single tool:

- **CPU single-core**: simplest "works almost everywhere" request
- **CPU multi-thread**: requires a valid PE + slot alignment
- **GPU**: requires site GPU keys and correct node classes
- **Watcher/daemon**: lightweight, long walltime, minimal resources
- **Transfer/download-only**: network-heavy, compute-light (site-specific)

## Step 3: Build resource requests using the capability map

### Safe defaults (when uncertain)

If PE/memory keys are unknown, prefer **minimal** requests:

- 1 core
- modest walltime (30-60 min)
- no special memory keys
- no GPUs

If the walltime key is unknown, rely on site default runtime and state that assumption explicitly.
Then escalate once the site's resource vocabulary is confirmed.

### Standard resource knobs (SGE concepts)

- walltime: `-l <walltime_key>=HH:MM:SS` (for example `h_rt`; confirm via capability probe/site docs)
- slots: `-pe <pe_name> <slots>` (requires PE name)
- memory: site-defined (e.g. `-l mem_free=...` or `-l h_vmem=...` or per-core variants)
- GPUs: site-defined (e.g. `-l gpu=1`, `-l gpus=1`, capability keys, etc.)

## Interactive sessions (portable)

### Interactive command selection

- Prefer `qrsh` when available.
- Else use `qlogin` if that's the site's interactive entrypoint.
- Else use site UI (e.g. OnDemand) if provided (site-specific; see references).

### Generic interactive template (fill from capability map)

```bash
# ACCOUNT_ARG may be empty if unknown / not required
# PE_ARG may be empty if PE unknown (then you are effectively single-core)
# WALLTIME_ARG may be empty if walltime key is unknown and site default is acceptable
<interactive_cmd> \
  <ACCOUNT_ARG> \
  <WALLTIME_ARG> \
  <PE_ARG> \
  <MEM_ARG> \
  -cwd \
  -now n
```

**Inside the interactive node**, print minimal provenance to the log/output:

```bash
hostname
pwd
date
echo "NSLOTS=${NSLOTS:-unset}"
echo "TMPDIR=${TMPDIR:-unset}"
```

## Batch submission (portable)

### Minimal robust script header (recommended)

```bash
#!/bin/bash -l
#$ -N <job_name>
#$ -cwd
#$ -j y
#$ -o outputs/logs/$JOB_NAME.$JOB_ID.out

# Optional / site-dependent directives:
#  - accounting/project: #$ -P <project>
#  - walltime:          #$ -l <walltime_key>=HH:MM:SS
#  - slots:             #$ -pe <pe_name> <slots>
#  - memory:            #$ -l <mem_key>=<value>
#  - GPUs:              #$ -l <gpu_key>=<value>

set -euo pipefail

echo "host: $(hostname)"
echo "workdir: $(pwd)"
echo "date: $(date)"
echo "JOB_ID=${JOB_ID:-unset}"
echo "NSLOTS=${NSLOTS:-unset}"
```

### Submit and capture the job id (prefer terse)

```bash
JOB_ID="$(qsub -terse <ACCOUNT_ARG> <script_or_template>)" || exit 1
echo "submitted JOB_ID=$JOB_ID"
```

If `-terse` is unsupported, capture stdout and parse the id from it.

## Monitoring and diagnosis (operations-friendly)

### Job state quick reference

- `qw`: queued/waiting
- `r`: running
- `Eqw`: error/hold state (investigate immediately)
- job disappears from `qstat`: finished or deleted (check logs/accounting)

### Core commands

```bash
qstat -u "$USER"
qstat -j "$JOB_ID"
```

### Eqw triage (portable)

```bash
qstat -j "$JOB_ID" | sed -n '1,200p'
```

Look for:

- invalid resource keys (`unknown resource`)
- invalid PE name
- missing paths / permissions for `-cwd` or `-o`
- rejected queue/request policies

## Watchers: bounded loops + teardown contract (mandatory)

**Why:** automation can accidentally leave infinite loops/subprocesses. This skill requires bounded watchers.

### Default watcher policy

Unless the user explicitly asks "watch until completion":

- queue watcher runs **until** job reaches `r` OR `Eqw` OR disappears OR **10 minutes**, then stops
- log watcher runs **until** the log file exists and shows first output OR **10 minutes**, then stops
- after stopping, report how to restart watchers on demand

### Queue watcher (bounded)

```bash
JOB_ID="<job_id>"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-600}"
INTERVAL_SEC="${INTERVAL_SEC:-20}"

elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT_SEC" ]; do
  line="$(qstat 2>/dev/null | awk -v id="$JOB_ID" '$1==id{print; exit}')"
  if [ -z "$line" ]; then
    echo "[info] job $JOB_ID not in qstat (finished or deleted)"
    break
  fi

  state="$(echo "$line" | awk '{print $5}')"
  echo "$(date) state=$state :: $line"

  if [ "$state" = "r" ] || echo "$state" | grep -q "E"; then
    echo "[info] watcher stopping (state=$state)"
    break
  fi

  sleep "$INTERVAL_SEC"
  elapsed=$((elapsed + INTERVAL_SEC))
done

if [ "$elapsed" -ge "$MAX_WAIT_SEC" ]; then
  echo "[info] watcher timed out after ${MAX_WAIT_SEC}s; re-run to continue monitoring"
fi
```

### Log watcher (bounded, no infinite tail)

```bash
LOG_PATH="outputs/logs/<job_name>.<job_id>.out"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-600}"
INTERVAL_SEC="${INTERVAL_SEC:-10}"

elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT_SEC" ]; do
  if [ -f "$LOG_PATH" ]; then
    echo "[info] log exists: $LOG_PATH"
    tail -n 60 "$LOG_PATH" || true
    break
  fi
  sleep "$INTERVAL_SEC"
  elapsed=$((elapsed + INTERVAL_SEC))
done

if [ "$elapsed" -ge "$MAX_WAIT_SEC" ]; then
  echo "[info] log did not appear after ${MAX_WAIT_SEC}s; check -cwd/-o paths or qstat -j output"
fi
```

### Teardown rule (execution contract)

If a background watcher subprocess is started, it must:

- record its PID in the run handle
- stop it when the watcher goal is reached or timeout triggers
- report "watchers stopped" (or "watchers still running, PID=...") explicitly

## "Step into the node" policy guardrails (do not assume SSH)

Attaching to a running batch job environment is not guaranteed.

Supported levels:

1. **Always:** log-based stepping (tail/read log files)
2. **Sometimes:** discover `exec_host` and SSH *if allowed by site policy*
3. **Best UX:** use interactive sessions for true "shell inside node" workflows

**Guardrails:**

- Do not suggest SSH into compute nodes unless policy is confirmed.
- If attempted, do a harmless probe first (`ssh <host> hostname`) and stop if denied.
- Prefer interactive sessions to avoid policy violations.

## Repo paths and templates: never hard-code

If referring to repo-local job templates or docs:

1. resolve repo root
2. discover candidate templates with `find`
3. print the resolved absolute path before submitting

Example:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
find "$REPO_ROOT" -maxdepth 6 -type f \( -name "*.qsub" -o -name "*.sge" -o -name "*.sh" \) | sed -n '1,120p'
```

If a template path is uncertain, do not guess; discover it.

## Progressive disclosure: site/workload references

This core skill stays portable. For specifics in this repository:

- BU SCC platform docs: `../README.md`
- BU SCC quickstart: `../quickstart.md`
- BU SCC install bootstrap: `../install.md`
- BU SCC batch + Notify runbook: `../batch-notify.md`
- BU SCC job templates: `../jobs/README.md`
- dnadesign workload examples: `references/workload-dnadesign.md`
