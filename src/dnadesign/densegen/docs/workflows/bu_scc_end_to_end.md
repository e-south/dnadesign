# BU SCC End-to-End: DenseGen -> USR -> Notify

This runbook is BU SCC specific.

Intent:
- run DenseGen on BU SCC with SGE/qsub
- store outputs in USR dataset layout
- watch USR `.events.log` with Notify
- sync datasets between SCC and local with USR remotes

Boundary contract:
- Notify input is USR `<dataset>/.events.log`
- DenseGen `outputs/meta/events.jsonl` is diagnostics only

---

## 1) Connect to SCC

```bash
ssh <BU_USERNAME>@scc1.bu.edu
# If your client needs it for Duo:
# ssh -o PasswordAuthentication=no <BU_USERNAME>@scc1.bu.edu
```

Reference:
- [BU SCC SSH access](https://www.bu.edu/tech/support/research/system-usage/connect-scc/ssh/)

OnDemand is also supported for shell/file workflows:
- [SCC OnDemand](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/)

---

## 2) Prepare workspace + config

Use project/scratch storage and keep the USR root in workspace `outputs/`:

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

## 3) Submit SGE job

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

CONFIG="/project/$USER/densegen_runs/bu_scc_demo/config.yaml"

export USR_ACTOR_TOOL=densegen
export USR_ACTOR_RUN_ID="${JOB_ID}_${SGE_TASK_ID:-0}"

uv run dense run --no-plot -c "$CONFIG"
```

Submit:

```bash
qsub run_densegen.sh
```

Reference:
- [BU SCC submitting jobs](https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/)
- [BU SCC batch script examples](https://www.bu.edu/tech/support/research/system-usage/running-jobs/batch-script-examples/)

---

## 4) Watch events with Notify

Run from login shell or OnDemand shell:

```bash
uv run notify profile wizard \
  --profile outputs/notify.profile.json \
  --provider slack \
  --events /project/$USER/densegen_runs/bu_scc_demo/outputs/usr_datasets/densegen/bu_scc_demo/.events.log \
  --secret-source auto \
  --preset densegen

uv run notify profile doctor --profile outputs/notify.profile.json
uv run notify usr-events watch --profile outputs/notify.profile.json --follow
```

---

## 5) Transfer datasets back to local

Configure remotes once on local:

```bash
export USR_REMOTES_PATH="$HOME/.config/dnadesign/usr-remotes.yaml"

uv run usr remotes wizard \
  --preset bu-scc \
  --name bu-scc \
  --user <BU_USERNAME> \
  --host scc1.bu.edu \
  --base-dir /project/<BU_USERNAME>/densegen_runs/bu_scc_demo/outputs/usr_datasets

uv run usr remotes doctor --remote bu-scc
uv run usr pull densegen/bu_scc_demo --remote bu-scc -y
```

For transfer-heavy jobs, BU provides `scc-globus.bu.edu` and download-node workflows (`qsub -l download`):
- [BU SCC transfer node docs](https://www.bu.edu/tech/support/research/system-usage/transferring-files/cloud-applications/)

---

## 6) Array job pattern

```bash
#!/bin/bash -l
#$ -N densegen_array
#$ -cwd
#$ -j y
#$ -o densegen_array.$JOB_ID.$TASK_ID.out
#$ -t 1-16

set -euo pipefail

CONFIG="/project/$USER/densegen_runs/run_${SGE_TASK_ID}/config.yaml"
export USR_ACTOR_TOOL=densegen
export USR_ACTOR_RUN_ID="${JOB_ID}_${SGE_TASK_ID}"

uv run dense run --no-plot -c "$CONFIG"
```
