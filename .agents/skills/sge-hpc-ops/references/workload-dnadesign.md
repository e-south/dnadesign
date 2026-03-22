## dnadesign Workload Profile

Use this reference only when working inside the `dnadesign` repository.

### Discover docs and templates

```bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
find "$REPO_ROOT/docs/bu-scc/jobs" -maxdepth 1 -type f -name "*.qsub" 2>/dev/null
```

Primary docs:
- `docs/bu-scc/quickstart.md`
- `docs/bu-scc/batch-notify.md`
- `docs/bu-scc/jobs/README.md`

### Storage and quota precheck

Run before large DenseGen submits:

```bash
groups
pquota
```

Placement guidance:
- keep hand-edited code and configs in repository paths under project storage
- place large generated artifacts under project disk (`/project` or `/projectnb`) rather than home
- use node-local scratch only for temporary runtime intermediates

### Stress ethanol and ciprofloxacin workspace

Workspace path:
- `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml`

Current packaged solver backend is `GUROBI` in this workspace.
Use this check to verify backend state before submit:

```bash
rg -n "backend:" src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml
```

### BU SCC DenseGen batch submit pattern

Preferred runbook-native Ops path:

```bash
uv run ops runbook precedents
uv run ops runbook init \
  --workflow densegen \
  --runbook <runbook.yaml> \
  --workspace-root src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro \
  --repo-root <repo> \
  --project <project> \
  --id <runbook-id> \
  --h-rt 02:00:00 --pe-omp 16 --mem-per-core 8G
```

DenseGen scaffolds include notify by default. Add `--no-notify` only when the request explicitly asks for batch-only submit.

Direct qsub pattern (explicit manual submit path):

```bash
qsub -P <project> \
  -pe omp 16 \
  -l h_rt=08:00:00 \
  -l mem_per_core=8G \
  -v DENSEGEN_CONFIG=<repo>/src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml \
  docs/bu-scc/jobs/densegen-cpu.qsub
```

### DenseGen + Notify Slack chained workflow

Use this path for requests like "start DenseGen and also wire Notify for Slack".

```bash
CONFIG=<repo>/src/dnadesign/densegen/workspaces/<workspace>/config.yaml
NOTIFY_DIR="<repo>/src/dnadesign/densegen/workspaces/<workspace>/outputs/notify/densegen"

uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source auto \
  --policy densegen

NOTIFY_JOB_ID="$(qsub -terse -P <project> \
  -v NOTIFY_PROFILE="$NOTIFY_DIR/profile.json" \
  docs/bu-scc/jobs/notify-watch.qsub)"

DENSEGEN_JOB_ID="$(qsub -terse -P <project> \
  -v DENSEGEN_CONFIG="$CONFIG" \
  docs/bu-scc/jobs/densegen-cpu.qsub)"

printf 'notify_job_id=%s\ndensegen_job_id=%s\n' "$NOTIFY_JOB_ID" "$DENSEGEN_JOB_ID"
```

Notes:
- `notify-watch.qsub` uses `--wait-for-events`, so watcher-first submit is valid.
- Notify watches USR `.events.log`, not DenseGen `outputs/meta/events.jsonl`.

### OnDemand session handoff in dnadesign context

When the user says they already entered OnDemand:
- skip session creation
- confirm locus and repo context quickly
- continue task actions from that shell or route long runs to batch

```bash
hostname
pwd
command -v qsub && qstat -u "$USER" | sed -n '1,60p'
```

### GUROBI variant

For `GUROBI` runs:
- keep `densegen.solver.threads <= requested slots`
- set solver and runtime caps in config before submit

### BU SCC transfer and connectivity guardrails

- For large data movement, route through BU transferring-files guidance before suggesting transfer-node jobs.
- For interactive reconnectability, route through BU OnDemand guidance before suggesting qrsh-only flows.

Solver backend defaults are workspace-specific; verify `config.yaml` before submit.
