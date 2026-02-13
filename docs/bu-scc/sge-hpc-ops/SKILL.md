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
- **Gate submissions.** Do not run real `qsub` until verify and runtime prechecks pass.
- **Re-probe when context changes.** If login host/site context changes, regenerate the capability map before new commands.
- **Use an explicit interactive state machine.** Never "try again" silently; every submit/reattach decision must be state-driven and user-visible.

## Operator objects to maintain

### 1) Capability map (mandatory)

A small mapping that records what *this* cluster supports and the correct resource key names.

Minimum fields (extend as needed):
- `accounting`: `{ supported: bool, flag: "-P" | "-A" | ..., notes }`
- `interactive_cmd`: `"qrsh"` or `"qlogin"` (or unsupported)
- `interactive_cmd_capabilities`: `{ qrsh: { supports: [...] }, qlogin: { supports: [...] }, notes }`
- `pe`: `{ supported: bool, flag: "-pe", names: [...], default_pe }`
- `walltime`: `{ supported: bool, key: "h_rt"|..., examples }`
- `memory`: `{ supported: bool, style: "per_core"|"total", key: "mem_per_core"|..., examples }`
- `gpu`: `{ supported: bool, keys: [...], examples }`
- `transfer`: `{ supported: bool, key_or_queue: "...", examples }`
- `lifecycle`: `{ qdel: bool, qmod_clear_eqw: "supported"|"unknown"|"unsupported", qacct: bool, notes }`
- `policy`: `{ ssh_to_compute_allowed: "unknown"|"yes"|"no", notes }`
- `probed`: `{ host, user, timestamp_utc }`

### 2) Run handle (per job/session)

- `mode`: `interactive_cli_qrsh` | `interactive_ondemand_duo` | `batch`
- `queue_safe`: `true|false` (default `true` for interactive)
- `intent`: `rejoinable` | `direct_exec` | `unknown`
- `job_id` (batch/qrsh) or `session_id` (OnDemand/UI tracking id if available)
- `job_name`
- `workdir`
- `submit_time_utc`
- `state` (`qw`/`r`/`Eqw`/`done`/`unknown`)
- `queue`
- `log_path` (stdout/stderr)
- `exec_host` (if discoverable)
- `fingerprint`: `{ account_or_project, pe_slots, mem_request, walltime_request, cwd, mode, intent }`
- `auth_state`: `not_required` | `pending_user_confirmation` | `pending_verification` | `verified` | `unknown`
- `recovery_attempts`: integer (for guarded retry accounting)
- `last_seen_utc`
- `watchers`: list of watcher subprocesses with **timeout + teardown plan**

### 3) Interactive request fingerprint (mandatory)

Define a request fingerprint before interactive submission:
- `account_or_project` (exact selected accounting value)
- `pe_slots` (PE name + slots, or `none`)
- `mem_request` (exact resource key/value tuple, or `none`)
- `walltime_request` (exact key/value tuple, or `none`)
- `cwd` (resolved absolute working directory)
- `mode` (`interactive_cli_qrsh` or `interactive_ondemand_duo`)
- `intent` (`rejoinable` or `direct_exec`)

Matching contract:
- A "matching interactive job" must match all non-empty fingerprint fields.
- If multiple matches exist, stop and ask user which handle to continue with.
- If no exact match exists, treat as non-match (do not reuse).

### 4) Interactive state machine (mandatory)

Use this finite-state flow for all interactive requests:

1. `route_intent`: choose mode from user intent:
   - `rejoinable` => `interactive_ondemand_duo`
   - `direct_exec` => `interactive_cli_qrsh`
2. `derive_fingerprint`: compute the full request fingerprint after mode/intent are known.
3. `detect_existing`: inspect `qstat -u "$USER"` and classify fingerprint-matching interactive jobs (`qw`/`r`/none).
4. `submit_once` (qrsh mode only): if no matching job exists, submit exactly one request with `-now n`.
5. `wait_bounded`: watch only the tracked job id (no new submissions), bounded by timeout.
6. `attach_or_handshake`:
   - qrsh mode: attach when state becomes `r`
   - OnDemand mode: pause for user confirmation after Duo
7. `post_handshake_verify` (OnDemand mode): set `auth_state=pending_verification` and require a verification check before claiming the session is active.
8. `interrupted_recover`: on next turn, always re-run `detect_existing` before any new submit.
9. `orphan_decision`: if orphaned/unusable, ask user `keep` vs `qdel`.
10. `guarded_retry_once`: allow one retry only when all conditions hold:
   - explicit reason is reported (for example client interruption, scheduler transient)
   - same fingerprint is reused
   - user confirms retry in this turn
   - `recovery_attempts` has not exceeded one
11. `closed`: after clean `exit`/completion, confirm with `qstat -u "$USER"`.

State-machine hard rules:
- Never submit a second interactive request by default.
- Use only `guarded_retry_once` for one retry under explicit conditions.
- Never try queue-bypass/priority tricks unless user explicitly asks.
- Always report the tracked handle (`mode`, `job_id/session_id`, `state`, `queue`, `exec_host`, `last_seen_utc`) in updates.

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

**Tie-break when both flags exist:**
- In this repository's BU SCC context, prefer `-P <project>`.
- Outside BU SCC, ask for site docs or a known submit example and do not emit both `-P` and `-A`.

### Account/project validity gate (hard-stop)

Do not run real `qsub` unless account/project validity is proven by at least one of:

1. user-provided known-good submit example using the same flag/value
2. prior successful job evidence for current user with the same flag/value
3. explicit site doc instruction plus user-confirmed value

If validity cannot be proven, stop and ask for a known-good example or corrected value.
Never silently switch between `-P` and `-A`. Never emit both.

### Probe E: lifecycle tooling detection (best effort)

Treat unsupported or denied commands as optional capability gaps, not hard failures.

```bash
command -v qdel >/dev/null && echo "qdel available" || true
command -v qmod >/dev/null && echo "qmod available" || true
command -v qacct >/dev/null && echo "qacct available" || true

qmod -help 2>&1 | grep -E -- '-cj|clear' || true
qacct -help 2>&1 | sed -n '1,60p' || true
```

### Probe F: interactive command option capabilities (mandatory for interactive commands)

Do not assume `qrsh` and `qlogin` share the same options. Build `interactive_cmd_capabilities` from help output.

```bash
qrsh -help 2>&1 | sed -n '1,120p' || true
qlogin -help 2>&1 | sed -n '1,120p' || true
```

## Step 0.5: Capability snapshot output (mandatory before command generation)

Before proposing interactive or batch commands, print a compact capability snapshot for the current host.

Template:

```text
Capability snapshot
- host: <hostname>
- scheduler_tools: qsub=<yes/no> qstat=<yes/no> qdel=<yes/no>
- interactive_cmd: <qrsh|qlogin|unknown>
- interactive_cmd_capabilities: qrsh=<supported|unknown> qlogin=<supported|unknown>
- accounting_flag: <none|-P|-A|both(choose -P for BU SCC)>
- pe_known: <yes/no> default_pe=<name|unknown>
- walltime_key: <h_rt|unknown>
- memory_key: <mem_per_core|mem_free|h_vmem|unknown>
- gpu_keys: <gpus,gpu_c,...|unknown>
- transfer_key: <download|unknown>
- lifecycle: qdel=<yes/no> qmod_cj=<yes/no> qacct=<yes/no>
- unknowns: <comma-separated unknown capability fields or "none">
```

If unknowns remain, call them out and use the safe-default flow.

## Step 1: Choose an execution mode (interactive vs batch)

### Choose interactive when:

- debugging environment issues
- iterating quickly
- you need a real shell "inside the node" as the primary workspace
- you need either:
  - agent-controlled command execution (`interactive_cli_qrsh`)
  - user-authenticated reconnectability (`interactive_ondemand_duo`)

### Choose batch when:

- runs are long
- you need restart safety + durable logs
- you want deterministic scheduler-managed resources
- you want to run watchers/daemons without keeping an interactive terminal open

### Interactive intent routing (mandatory)

Route interactive requests by stated intent before generating commands:

- If user says "must be rejoinable", "browser/portal is fine", or equivalent:
  - route to `interactive_ondemand_duo`
  - do not default to plain `qrsh`
- If user says agent must execute commands directly in the allocation:
  - route to `interactive_cli_qrsh`
  - explicitly warn this path is less reconnectable after client interruption
- If intent is unclear:
  - ask one short routing question before submitting anything

## Step 2: Pick a workload pattern (portable categories)

Use these categories; do not tie the core logic to any single tool:

- **CPU single-core**: simplest "works almost everywhere" request
- **CPU multi-thread**: requires a valid PE + slot alignment
- **GPU**: requires site GPU keys and correct node classes
- **Watcher/daemon**: lightweight, long walltime, minimal resources
- **Transfer/download-only**: network-heavy, compute-light (site-specific)

## Step 3: Build resource requests using the capability map

This skill covers requesting scheduler resources for jobs. It does not create scheduler objects such as queues, complexes, projects, or host groups.

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

### Interactive mode summary

| User intent | Mode | Authentication path | Reconnectability | Primary control |
|---|---|---|---|---|
| agent executes commands directly in allocation | `interactive_cli_qrsh` | scheduler CLI (`qrsh`/`qlogin`) | lower (client interruption can break attach) | agent |
| session should be rejoinable | `interactive_ondemand_duo` | browser + Duo via OnDemand | higher (portal-managed sessions) | user + portal |

### Queue-safe policy (`queue_safe=true`, default)

Before any interactive submit:

```bash
qstat -u "$USER"
```

Rules:
- If a fingerprint-matching interactive request is already `qw`/`r`, reuse/watch it; do not submit another.
- If none exists and mode is `interactive_cli_qrsh`, submit exactly one interactive request.
- If mode is `interactive_ondemand_duo`, run handshake + verification instead of scheduler submit.
- Use `-now n` and bounded polling (default 20s interval, 600s max) for queue patience.
- Never resubmit automatically on timeout.
- If timeout/recovery is needed, use `guarded_retry_once` with explicit reason + same fingerprint + user confirmation.

### Mode A: `interactive_cli_qrsh` (agent-controlled, less reconnectable)

Use when the user wants the agent to execute commands directly in the interactive allocation.

Command template:

```bash
# Build flags from interactive_cmd_capabilities; do not assume qlogin/qrsh parity.
# ACCOUNT_ARG may be empty if unknown / not required.
# PE_ARG may be empty if PE unknown (then you are effectively single-core).
# WALLTIME_ARG may be empty if walltime key is unknown and site default is acceptable.
<interactive_cmd> <supported_flags_only> <ACCOUNT_ARG> <WALLTIME_ARG> <PE_ARG> <MEM_ARG>
```

After attach, print minimal provenance:

```bash
hostname
pwd
date
echo "NSLOTS=${NSLOTS:-unset}"
echo "TMPDIR=${TMPDIR:-unset}"
```

Recovery note:
- if the controlling client is interrupted, this mode may lose attachability; treat recovery as `detect_existing` + explicit `keep|qdel` decision.

### Mode B: `interactive_ondemand_duo` (human-authenticated, reconnect-friendly)

Use when the user prioritizes reconnectability and accepts browser-based authentication.

Handshake contract:
1. Agent provides OnDemand URL and exact next steps.
2. User performs login + Duo in browser.
3. Agent pauses and waits for explicit user confirmation:
   - `Duo approved and session started`
4. Agent sets `auth_state=pending_verification` and runs verification checks.
5. Agent proceeds only after verification succeeds (`auth_state=verified`), else reports `unverified` and asks for next action.

Recommended BU endpoint:
- `https://scc-ondemand.bu.edu/`

Important constraints:
- agent cannot complete Duo itself
- agent cannot inherit browser-authenticated session cookies
- agent must wait for user confirmation before proceeding with OnDemand-dependent actions

#### OnDemand verification rubric (mandatory)

Purpose:
- prevent false-positive "session started" claims
- preserve queue discipline (no hidden replacement submits)
- keep user interaction explicit and low-friction

Verification states:
- `pending_user_confirmation`: waiting for user to complete Duo/login
- `pending_verification`: user confirmed, evidence threshold not yet met
- `verified`: evidence threshold met
- `unverified`: verification failed or timed out
- `unknown`: evidence is partial or ambiguous

Evidence sources:
- `E1_USER_CONFIRM` (required): user states `Duo approved and session started`
- `E2_PORTAL_SESSION`: OnDemand session metadata (session/app id, app type, start time, state)
- `E3_SCHEDULER_MATCH`: `qstat -u "$USER"` shows a job consistent with fingerprint and time window
- `E4_EXEC_PROOF`: user provides in-session proof (`hostname`, `pwd`, `date`)

Verification threshold:
- set `auth_state=verified` only when all conditions hold:
  - `E1_USER_CONFIRM` is present
  - at least one of `E2_PORTAL_SESSION` or `E3_SCHEDULER_MATCH` is present
  - if multiple candidate sessions exist, require `E4_EXEC_PROOF` or explicit user selection of a session id
- otherwise keep `auth_state=pending_verification` or set `auth_state=unknown`

Failure codes:
- `VFY-001`: user confirmation missing
- `VFY-002`: no portal/scheduler evidence after confirmation
- `VFY-003`: evidence does not match fingerprint (project/resources/cwd/mode/intent)
- `VFY-004`: multiple candidate sessions; identity unresolved
- `VFY-005`: verification timeout reached

UX contract:
- ask for one action per turn
- always show a compact status card:
  - `mode`, `auth_state`, `fingerprint_summary`, `candidate_sessions`, `next_required_action`
- use bounded waits for verification polling (default 20s interval, 600s max)
- never claim active session ownership before `auth_state=verified`
- never auto-submit replacement interactive requests while verification is unresolved

User prompts:
- auth prompt: `Please complete Duo and start the OnDemand session. Reply: done.`
- identity prompt: `I found multiple candidates. Reply with session number: 1 or 2.`
- proof prompt: `Run hostname; pwd; date in the new session and paste output.`
- timeout prompt: `Verification timed out (VFY-005). Reply: retry verify, keep waiting, or cancel.`

Post-confirmation verification rules:
- Use best-effort evidence from portal metadata, scheduler visibility, and user-provided session proof.
- Treat verification as successful only when threshold conditions are satisfied and attributable to this run handle.
- If evidence is missing, keep `auth_state=pending_verification`, report failure code, and request one concrete next action.

### Interruption recovery (mandatory)

At the start of every turn that touches interactive state:

```bash
qstat -u "$USER"
```

Then:
- if tracked job exists (`qw`/`r`), continue watching/using that handle
- if no tracked job exists, set interactive status to `ended_or_unknown`
- if job exists but appears unusable/orphaned, ask user: `keep` or `qdel <job_id>`

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

### Submit gate (mandatory before real `qsub`)

Run this gate in order and stop on first failure:

1. **Account/project validity proven**
- apply the hard-stop validity gate above

2. **Template path resolved and trusted**
- print the absolute path
- verify it points to an expected job template location

3. **Scheduler verify pass**

```bash
qsub -verify <ACCOUNT_ARG> "<template_or_script>"
```

4. **Runtime input prechecks pass**
- execute the per-template matrix below
- output/log directory path is writable

5. **Only then submit**

```bash
JOB_ID="$(qsub -terse <ACCOUNT_ARG> "<template_or_script>")" || exit 1
echo "submitted JOB_ID=$JOB_ID"
```

`qsub -verify` is necessary but not sufficient. It validates scheduler syntax, not runtime input validity.

`qsub -verify` metadata can be misleading in some contexts (for example blank `owner` or `uid: 0` in dry runs).
Do not treat verify metadata fields as proof of runtime readiness or account validity.

### Per-template runtime precheck matrix (hard-stop)

| Template | Required inputs | Validation checks | Hard-stop condition |
|---|---|---|---|
| `docs/bu-scc/jobs/densegen-cpu.qsub` | proven-valid `ACCOUNT_ARG`; readable `DENSEGEN_CONFIG` | template path exists; `qsub -verify`; `test -r "$DENSEGEN_CONFIG"`; output dir writable | any check fails |
| `docs/bu-scc/jobs/evo2-gpu-infer.qsub` | proven-valid `ACCOUNT_ARG`; GPU keys confirmed in capability map; module strategy confirmed (`CUDA_MODULE`/`GCC_MODULE` or preloaded modules) | template path exists; `qsub -verify`; capability map includes GPU request keys; module command availability confirmed | any check fails |
| `docs/bu-scc/jobs/notify-watch.qsub` | proven-valid `ACCOUNT_ARG`; either `NOTIFY_PROFILE` path or explicit env mode (`NOTIFY_TOOL`,`NOTIFY_CONFIG`,`WEBHOOK_ENV` and policy/namespace inputs) | template path exists; `qsub -verify`; required file/env checks pass; output dir writable | any check fails |

Execution rule:
- if any required check is `FAIL`, stop immediately and report exactly what is missing
- do not submit while any check is unresolved

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

## Session and lifecycle controls (portable)

### Check active sessions and jobs

Use these checks before starting new work so existing runs are not duplicated:

```bash
# Active jobs for current user
qstat -u "$USER"

# Detailed view for one job
qstat -j "$JOB_ID"

# Best-effort filter for interactive-style jobs (site naming may differ)
qstat -u "$USER" | grep -E 'qrsh|qlogin|interact' || true
```

Interpretation rule:
- if `qstat -u "$USER"` returns no job rows and no command error, treat this as "no active jobs for this user", not as a probe failure.
- name-based filters are advisory only; do not use them as sole proof of absence/presence.

For web portal sessions (for example OnDemand), include a best-effort name filter and then use platform docs for authoritative session control:

```bash
qstat -u "$USER" | grep -E 'ood-|ood_|ondemand|jupyter|rstudio|desktop' || true
```

Ambiguity contract (interactive-sensitive tasks):
- if `qstat -u "$USER"` is empty and session status matters, set `session_status=unknown` (not `none`)
- consult BU OnDemand session docs via `../README.md`
- ask the user to confirm active portal sessions before launching a new interactive session
- by default, pause launch while status is unresolved
- if user explicitly requests to proceed despite unresolved status, allow one launch attempt with full risk note and fingerprint tracking
- do not submit replacement interactive requests automatically while status is unresolved

### Start and stop gracefully

Start:
- interactive (agent-controlled): launch `qrsh`/`qlogin` per capability map and `queue_safe=true`
- interactive (rejoinable): use OnDemand + Duo handshake, then require post-confirmation verification before continuing
- batch: submit with `qsub` and capture job id

Stop:
- interactive shell: exit cleanly with `exit`
- batch job: `qdel "$JOB_ID"`
- background watchers: stop by PID per teardown contract

### Lifecycle safety gate (mandatory before `qdel` or `qmod -cj`)

Do not mutate job state unless all checks pass:

1. inspect `qstat -j "$JOB_ID"` and capture triage reason
2. confirm job owner matches current user before action
3. for `qmod -cj`, use only for transient/recoverable causes
4. allow at most one clear-and-recheck cycle before requiring deeper diagnosis

If owner cannot be confirmed from scheduler output, stop and ask the user before mutating job state.

Owner check helper:

```bash
owner_qstat="$(qstat 2>/dev/null | awk -v id="$JOB_ID" '$1==id || index($1, id ".")==1 {print $4; exit}')"
owner_detail="$(qstat -j "$JOB_ID" 2>/dev/null | awk -F': *' '/^owner:/{print $2; exit}')"

owner="${owner_qstat:-$owner_detail}"
if [ -z "$owner" ]; then
  echo "[stop] unable to confirm owner for JOB_ID=$JOB_ID"
  exit 2
fi

echo "owner=$owner user=$USER"
test "$owner" = "$USER"
```

### Optional Eqw recovery path

Use only after triage confirms a recoverable transient issue and site policy allows it.

```bash
qstat -j "$JOB_ID" | sed -n '1,200p'
qmod -cj "$JOB_ID" || true
qstat -j "$JOB_ID" | sed -n '1,120p'
```

If `qmod` is unavailable or denied, do not force retries; fix request errors and resubmit.

### Optional postmortem accounting path

When a job has finished and accounting is enabled:

```bash
qacct -j "$JOB_ID" | sed -n '1,200p' || true
```

If `qacct` is unavailable, use scheduler output plus job logs as the postmortem source.

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
  rows="$(qstat 2>/dev/null | awk -v id="$JOB_ID" '$1==id || index($1, id ".")==1')"
  if [ -z "$rows" ]; then
    echo "[info] job $JOB_ID not in qstat (finished or deleted)"
    break
  fi

  total="$(echo "$rows" | wc -l | tr -d ' ')"
  running="$(echo "$rows" | awk '$5=="r"{c++} END{print c+0}')"
  pending="$(echo "$rows" | awk '$5=="qw" || $5=="hqw"{c++} END{print c+0}')"
  errors="$(echo "$rows" | awk '$5 ~ /E/{c++} END{print c+0}')"

  echo "$(date) total=$total running=$running pending=$pending errors=$errors"

  if [ "$errors" -gt 0 ]; then
    echo "[info] watcher stopping (error state present)"
    break
  fi

  if [ "$running" -gt 0 ] && [ "$pending" -eq 0 ]; then
    echo "[info] watcher stopping (all visible tasks running)"
    break
  fi

  if [ "$running" -gt 0 ] && [ "$pending" -gt 0 ]; then
    echo "[info] mixed task states (running+pending); continuing until timeout/error/all-running"
  fi

  if [ "$running" -eq 0 ] && [ "$pending" -eq 0 ]; then
    echo "[info] no running/pending tasks visible; keep monitoring briefly or inspect qstat -j"
  fi

  sleep "$INTERVAL_SEC"
  elapsed=$((elapsed + INTERVAL_SEC))
done

if [ "$elapsed" -ge "$MAX_WAIT_SEC" ]; then
  echo "[info] watcher timed out after ${MAX_WAIT_SEC}s; re-run to continue monitoring"
fi
```

Array note:
- some sites display array tasks as `<job_id>.<task_id>` or separate task columns.
- this watcher aggregates states across matching tasks and does not rely on a first row.

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

# Prefer canonical BU SCC templates first in this repository
if [ -d "$REPO_ROOT/docs/bu-scc/jobs" ]; then
  find "$REPO_ROOT/docs/bu-scc/jobs" -maxdepth 1 -type f -name "*.qsub" | sed -n '1,120p'
fi

# Fallback discovery (exclude common non-source/vendor dirs)
find "$REPO_ROOT" -maxdepth 8 -type f \
  \( -name "*.qsub" -o -name "*.sge" \) \
  -not -path "*/.venv/*" \
  -not -path "*/.git/*" \
  -not -path "*/node_modules/*" \
  -not -path "*/.mypy_cache/*" \
  -not -path "*/.pytest_cache/*" \
  | sed -n '1,200p'

# Optional shell harness discovery (explicit opt-in only)
if [ "${ALLOW_SHELL_HARNESS:-0}" = "1" ]; then
  find "$REPO_ROOT" -maxdepth 8 -type f -name "*.sh" \
    -not -path "*/.venv/*" \
    -not -path "*/.git/*" \
    -not -path "*/node_modules/*" \
    -not -path "*/.mypy_cache/*" \
    -not -path "*/.pytest_cache/*" \
    | sed -n '1,120p'
fi
```

If a template path is uncertain, do not guess; discover it.

When selecting a template for submission, use `.qsub`/`.sge` files by default.
Use `.sh` only when the task explicitly requests a shell harness and `ALLOW_SHELL_HARNESS=1` is set.

## Progressive disclosure: site/workload references

This core skill stays portable. For specifics in this repository:

- BU SCC platform docs: `../README.md`
- BU SCC quickstart: `../quickstart.md`
- BU SCC install bootstrap: `../install.md`
- BU SCC batch + Notify runbook: `../batch-notify.md`
- BU SCC job templates: `../jobs/README.md`
- dnadesign workload examples: `references/workload-dnadesign.md`

## Additional resources discovery (site-agnostic pattern)

Use this order when more detail is needed than the core skill provides:

1. local platform docs in the repository
2. official site scheduler docs on the web
3. official site interactive portal/session docs on the web

For this repository, start with `../README.md` and related `../` BU SCC docs.
BU OnDemand discovery is documented in BU SCC links, not duplicated in this core skill.

## User-facing response contract (mandatory UX output)

When executing this skill against a real cluster, return concise didactic updates in this order:

1. **Findings:** confirmed facts from commands (tools, keys, states, job id, errors)
2. **Interpretation:** what those facts imply for next action
3. **Action:** exact next command(s) to run
4. **Risk/unknowns:** what remains unconfirmed and how to resolve it

Keep updates brief, specific, and operational. Do not return raw command dumps without interpretation.

Interactive session reporting requirements:
- Always include current run handle fields (`mode`, `job_id/session_id`, `state`, `queue`, `exec_host`, `last_seen_utc`).
- Always include request fingerprint summary (`account_or_project`, `pe_slots`, `mem_request`, `walltime_request`, `cwd`, `mode`, `intent`).
- When Duo/user auth is required, set status to `pending_user_confirmation`, then `pending_verification`, and proceed only after `verified`.
