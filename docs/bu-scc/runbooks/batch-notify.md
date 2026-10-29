## BU SCC Batch + Notify Runbook (`dnadesign` platform workflows)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-29

### Purpose

**Intent:** Run `dnadesign` workflows as SGE batch jobs on BU SCC with restart-safe logs and optional Notify webhooks.

**When to use:**
- Any run that should not execute in an interactive shell.
- Any run that needs explicit runtime/resource control.
- Any run that needs event-driven webhook notifications.


Related docs:
- Install/bootstrap: [BU SCC Install bootstrap](../setup/install.md)
- End-to-end quick path: [BU SCC Quickstart](../setup/quickstart.md)
- Job templates: [BU SCC jobs README](../jobs/README.md)
- Notify operator manual: [Notify SeqLedger events operator manual](../../notify/seqledger-events.md)

---

### 1) Scheduler rules that matter on BU SCC

BU SCC uses SGE (`qsub` + `#$` directives).

Operational rules to keep in mind:
- Always request runtime and resources explicitly for reproducible scheduling.
- If `h_rt` is omitted, BU defaults jobs to a 12-hour walltime.
- Runtime ceilings differ by job class (for example: single/OMP up to 720h, MPI up to 120h, GPU up to 48h).
- `-pe omp N` requests CPU slots for OpenMP-style jobs; keep solver thread counts aligned with `N`.

BU references:
- Submitting jobs: <https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/>
- Interactive jobs: <https://www.bu.edu/tech/support/research/system-usage/running-jobs/interactive-jobs/>
- Batch reference: <https://www.bu.edu/tech/support/research/system-usage/running-jobs/advanced-batch/>

### 1.1) Queue-fair pressure gate before new submits

Run this check before proposing additional `qsub` calls:

```bash
qstat -u "$USER" | awk '
  $1 ~ /^[0-9]+$/ {
    state=$5
    if (state ~ /r/) running++
    if (state ~ /q/) queued++
    if (state ~ /Eqw/) eqw++
  }
  END { printf "running_jobs=%d queued_jobs=%d eqw_jobs=%d\n", running, queued, eqw }
'
```

Policy:
- if `running_jobs > 3`, warn before additional submissions and confirm intent
- use `qsub -t` arrays for independent multi-batch fanout
- use `-hold_jid` chains for ordered multi-stage flows
- respect the queue and do not skip the line

---

### 2) Minimal `qsub` template (SGE)

```bash
#!/bin/bash -l
#$ -P <project>
#$ -N <job_name>
#$ -l h_rt=04:00:00
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -j y
#$ -o outputs/logs/$JOB_NAME.$JOB_ID.out

set -euo pipefail
```

---

### 3) Canonical job scripts

Use versioned templates from [BU SCC jobs README](../jobs/README.md):
- [Evo2 GPU template](../jobs/evo2-gpu-infer.qsub)
- [Permuter evaluate template](../jobs/permuter-evaluate.qsub)
- [Notify watcher template](../jobs/notify-watch.qsub)

Before first Evo2 submit on a host, run:
- [BU SCC install GPU setup and verification runbook](../setup/install.md#gpu-setup-and-verification-runbook)
- [infer SCC Evo2 GPU environment runbook](../../../src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md)

Submit examples:

```bash
qsub -P <project> \
  -v INFER_CONFIG=<dnadesign_repo>/src/dnadesign/infer/workspaces/<workspace>/config.yaml \
  docs/bu-scc/jobs/evo2-gpu-infer.qsub
qsub -P <project> \
  -v PERMUTER_WORKSPACE=<dnadesign_repo>/src/dnadesign/permuter/workspaces/<workspace>/config.yaml,PERMUTER_REF=<ref_name>,PERMUTER_EVALUATE_ARGS='--with smoke:placeholder:log_likelihood' \
  docs/bu-scc/jobs/permuter-evaluate.qsub
qsub -P <project> docs/bu-scc/jobs/notify-watch.qsub
```

---

### 4) Job arrays (parameter sweeps)

This scheduler skeleton selects one configuration per task. Add the owning
tool's command and resource requests before submission.

```bash
#!/bin/bash -l
#$ -P <project>
#$ -N <array_job>
#$ -t 1-32
#$ -l h_rt=04:00:00
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -j y
#$ -o outputs/logs/$JOB_NAME.$JOB_ID.$TASK_ID.out

set -euo pipefail

TASK_ID="${SGE_TASK_ID}"
CONFIG="/project/<project>/<user>/runs/config_${TASK_ID}.yaml"
```

---

### 5) Notify deployment patterns (BU SCC-safe)

#### Mode A (recommended): dedicated watcher batch job

Use a lightweight watcher job with profile-driven wiring.
This keeps webhook source, filters, cursor, and spool settings decoupled from submit commands.

Template submit example:

```bash
CONFIG=<dnadesign_repo>/src/dnadesign/infer/workspaces/<workspace>/config.yaml
NOTIFY_DIR="<dnadesign_repo>/src/dnadesign/infer/workspaces/<workspace>/outputs/notify/infer"
WEBHOOK_FILE="$HOME/.config/dnadesign/notify_webhook.secret"

mkdir -p "$(dirname "$WEBHOOK_FILE")"
touch "$WEBHOOK_FILE"
chmod 600 "$WEBHOOK_FILE"
uv run notify setup webhook \
  --secret-source file \
  --secret-ref "file://$WEBHOOK_FILE"

uv run notify setup slack \
  --tool infer \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source file \
  --secret-ref "file://$WEBHOOK_FILE" \
  --no-store-webhook \
  --policy infer

qsub -P <project> \
  -v NOTIFY_PROFILE="$NOTIFY_DIR/profile.json",WEBHOOK_FILE="$WEBHOOK_FILE" \
  docs/bu-scc/jobs/notify-watch.qsub
```

#### Mode B: explicit env wiring (no profile)

Use this when you intentionally do not want a profile file.
If `EVENTS_PATH` is omitted, the script resolves it from `NOTIFY_TOOL` and `NOTIFY_CONFIG`.
If `EVENTS_PATH` is explicit, set `NOTIFY_POLICY` (`infer` or `generic`)
and `NOTIFY_NAMESPACE` (for example, `infer`).

```bash
qsub -P <project> \
  -v NOTIFY_TOOL=infer,NOTIFY_CONFIG=<dnadesign_repo>/src/dnadesign/infer/workspaces/<workspace>/config.yaml,WEBHOOK_ENV=NOTIFY_WEBHOOK,WEBHOOK_FILE="$WEBHOOK_FILE" \
  docs/bu-scc/jobs/notify-watch.qsub
```

#### Mode C: short-lived validation in login/OnDemand shell

Use this only for setup checks (`profile doctor`, `--dry-run`) or troubleshooting.
For durable delivery, keep watchers in batch.

---

### 6) Large downloads and dataset/model transfer

Recommended pattern:
- Use BU Data Transfer Node for large model prefetch and large SeqLedger dataset push/pull.
- Keep compute-heavy work off transfer-only nodes.

Example transfer-node request:

```bash
qsub -l download <<'QSUB'
#!/bin/bash -l
#$ -P <project>
#$ -N transfer_job
#$ -l h_rt=24:00:00
#$ -pe omp 1
#$ -j y
#$ -o outputs/logs/transfer.$JOB_ID.out

set -euo pipefail

# Example: transfer-only operations
# uv run python scripts/prefetch_models.py
# uv run seqledger pull inference_demo bu-scc -y
QSUB
```

Transfer reference:
- <https://www.bu.edu/tech/support/research/system-usage/transferring-files/cloud-applications/>

---

### 7) Package boundary reminders

- Notify input: SeqLedger `<dataset>/.events.log`


---

Back: [BU SCC index](../README.md)

Next: [BU SCC Quickstart](../setup/quickstart.md)

Next: [BU SCC job templates](../jobs/README.md)
