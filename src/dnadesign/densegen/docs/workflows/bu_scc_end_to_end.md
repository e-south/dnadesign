# BU SCC End-to-End: DenseGen -> USR -> Notify

This runbook is BU SCC specific.

Intent:
- run DenseGen with SGE (`qsub`)
- write outputs to USR dataset layout
- watch USR `.events.log` with Notify
- sync datasets between SCC and local using USR remotes

Platform docs:
- [docs/hpc/bu_scc_install.md](../../../../../docs/hpc/bu_scc_install.md)
- [docs/hpc/bu_scc_batch_notify.md](../../../../../docs/hpc/bu_scc_batch_notify.md)

## Boundary contract

- Notify input: USR `<dataset>/.events.log`
- DenseGen `outputs/meta/events.jsonl`: diagnostics only

---

## 1) Connect to SCC

```bash
# Standard SCC login.
ssh <BU_USERNAME>@scc1.bu.edu

# Optional Duo-friendly variant (if your client needs it).
# ssh -o PasswordAuthentication=no <BU_USERNAME>@scc1.bu.edu
```

References:
- [BU SCC SSH access](https://www.bu.edu/tech/support/research/system-usage/connect-scc/ssh/)
- [SCC OnDemand](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/)

---

## 2) Prepare workspace config

Keep USR root inside workspace `outputs/`:

```yaml
densegen:
  run:
    id: bu_scc_demo
    root: "."
  output:
    targets: [usr]
    usr:
      root: outputs/usr_datasets
      dataset: densegen/bu_scc_demo
      chunk_size: 128
      health_event_interval_seconds: 60
```

---

## 3) Submit DenseGen SGE job

`run_densegen.sh`:

```bash
#!/bin/bash -l
#$ -N densegen_demo
#$ -cwd
#$ -j y
#$ -o densegen_demo.$JOB_ID.out
#$ -l h_rt=04:00:00
#$ -pe omp 4

set -euo pipefail

# Point to workspace config.
CONFIG="/project/$USER/densegen_runs/bu_scc_demo/config.yaml"

# Validate config + solver.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Run generation in batch mode.
uv run dense run --no-plot -c "$CONFIG"
```

Submit:

```bash
# Submit job to SGE queue.
qsub run_densegen.sh
```

---

## 4) Inspect run and resolve USR event path

```bash
# Show run summary with events and library details.
uv run dense inspect run --events --library -c /project/$USER/densegen_runs/bu_scc_demo/config.yaml

# Print exact USR .events.log path for Notify.
uv run dense inspect run --usr-events-path -c /project/$USER/densegen_runs/bu_scc_demo/config.yaml
```

---

## 5) Configure Notify watcher

```bash
# Export webhook URL for Notify.
export DENSEGEN_WEBHOOK="https://example.com/webhook"

# Create profile from USR events path.
uv run notify profile wizard \
  --profile /project/$USER/densegen_runs/bu_scc_demo/outputs/notify.profile.json \
  --provider slack \
  --events /project/$USER/densegen_runs/bu_scc_demo/outputs/usr_datasets/densegen/bu_scc_demo/.events.log \
  --cursor /project/$USER/densegen_runs/bu_scc_demo/outputs/notify.cursor \
  --spool-dir /project/$USER/densegen_runs/bu_scc_demo/outputs/notify_spool \
  --secret-source env \
  --url-env DENSEGEN_WEBHOOK \
  --only-tools densegen \
  --only-actions densegen_health,densegen_flush_failed,materialize

# Validate profile.
uv run notify profile doctor --profile /project/$USER/densegen_runs/bu_scc_demo/outputs/notify.profile.json

# Preview without delivery.
uv run notify usr-events watch --profile /project/$USER/densegen_runs/bu_scc_demo/outputs/notify.profile.json --dry-run

# Run live watcher.
uv run notify usr-events watch --profile /project/$USER/densegen_runs/bu_scc_demo/outputs/notify.profile.json --follow
```

---

## 6) Sync USR dataset between SCC and local

Run this on the machine where you want files to end up.

```bash
# Point USR to your remotes config.
export USR_REMOTES_PATH="$HOME/.config/dnadesign/usr-remotes.yaml"

# Create/check BU SCC remote.
uv run usr remotes wizard --preset bu-scc --name bu-scc --user <BU_USERNAME> --base-dir /project/<BU_USERNAME>/densegen_runs/outputs/usr_datasets
uv run usr remotes doctor --remote bu-scc

# Preview differences.
uv run usr diff densegen/bu_scc_demo bu-scc

# Pull from SCC to local.
uv run usr pull densegen/bu_scc_demo bu-scc -y
```

For deeper sync details, see:
- [../../../usr/docs/operations/sync.md](../../../usr/docs/operations/sync.md)

---

@e-south
