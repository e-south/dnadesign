## Batch Submit Contract

Enforce this gate before any real batch submit.

### Gate sequence

1. account or project validity is proven
2. template path is resolved and readable
3. submit script initialization is valid for module usage (use `#!/bin/bash -l` when modules are required)
4. output and scratch placement is valid for workload footprint (`/project` or `/projectnb` by default)
5. workload-shape checks pass (array, threads, memory, GPU)
6. scheduler syntax check passes
7. policy QA preflight check passes
8. session status check runs and pressure warning is acknowledged when running_jobs > 3
9. submission-shape advisor check passes and queue fairness policy is acknowledged
10. operator brief output is captured for user-facing submit gate reporting
11. runtime input prechecks pass
12. then submit and capture job id

### Scheduler syntax check

Prefer `qsub -verify` when available; fallback to `qsub -w v`.

```bash
if qsub -help 2>&1 | grep -q -- '-verify'; then
  qsub -verify <ACCOUNT_ARG> <TEMPLATE>
else
  qsub -w v <ACCOUNT_ARG> <TEMPLATE>
fi
```

### Session status and active-job pressure

Run this before each new submit group:

```bash
scripts/sge-session-status.sh --warn-over-running 3
```

Use this when a user asks for concrete active-job status before confirmation:

```bash
scripts/sge-active-jobs.sh --max-jobs 12
```

Rules:
- if `running_jobs > 3` and the user requests more submissions, warn before emitting `qsub` commands
- prefer array conversion or `-hold_jid` dependencies over unconstrained parallel submit expansion
- require explicit confirmation before proposing additional parallel submits over threshold

### Submission-shape advisor

Run advisor before proposing multi-submit plans:

```bash
scripts/sge-submit-shape-advisor.sh \
  --planned-submits <N> \
  --warn-over-running 3
```

Add `--requires-order` when dependency ordering is required.

Queue fairness:
- respect the queue and do not skip the line
- do not propose queue-bypass behavior
- avoid burst submit waves when advisor recommends `array` or `hold_jid`
- reject `-now y` in automated batch templates

### Operator brief

Run this when communicating submit readiness to the user:

```bash
scripts/sge-operator-brief.sh \
  --planned-submits <N> \
  --warn-over-running 3
```

### Workload-shape checks

- many similar tasks:
  - use array jobs (`-t start-end[:step]`) instead of many independent submissions
  - verify task logic references `SGE_TASK_ID`
- multi-job workflows:
  - prefer explicit dependencies (`-hold_jid`) for sequencing
  - avoid unbounded submit fanout in one turn
- threaded OpenMP-like jobs:
  - ensure requested slots and runtime thread count are aligned (`OMP_NUM_THREADS=$NSLOTS` when applicable)
- memory-heavy jobs:
  - set `mem_per_core` based on observed `maxvmem` and requested slots
  - validate large-memory requests against current BU examples before submit
- GPU jobs:
  - request GPU resources explicitly and verify the workload actually uses assigned GPUs

### Policy QA preflight

Run policy QA checks on submit artifacts before real submission.

```bash
scripts/qa-sge-submit-preflight.sh \
  --template <TEMPLATE> \
  --require-project-flag
```

### Runtime precheck examples

- DenseGen CPU template:
  - readable `DENSEGEN_CONFIG`
  - `uv run dense validate-config --probe-solver -c "$DENSEGEN_CONFIG"`

- Notify watcher template:
  - `NOTIFY_PROFILE` exists, or resolver inputs (`NOTIFY_TOOL` + `NOTIFY_CONFIG`) are present
  - webhook env var is set

- Transfer-node batch template (`-l download`):
  - treat as batch-oriented flow
  - do not use `qsh` or `qrsh` with `-l download`

### Process-reaper safety checks

- do not run CPU-intensive tasks on login nodes
- ensure runtime processes do not exceed requested slots
- for GPU jobs, stay within assigned devices and avoid idle-GPU timeouts
- when a reaper event occurs, capture evidence (`qstat`, job logs, scheduler mail) before any retry

### BU SCC policy handling

- cite BU running-jobs docs for scheduler lifecycle claims
- cite BU best-practices docs for shared-cluster usage recommendations
- cite BU advanced-batch docs for dependency and array semantics
- cite BU allocating-memory and resources-jobs docs for memory/resource claims
- cite BU process-reaper docs when recommending recovery after forced termination
- cite BU transferring-files docs before recommending transfer-node usage
- avoid hard-coded queue/resource policy claims without source-evidence entries

### Submit

```bash
JOB_ID="$(qsub -terse <ACCOUNT_ARG> <TEMPLATE>)" || exit 1
echo "submitted JOB_ID=$JOB_ID"
```
