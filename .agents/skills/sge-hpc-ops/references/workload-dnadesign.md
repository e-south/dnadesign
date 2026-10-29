## dnadesign Workload Profile

Use this reference only when working inside the `dnadesign` repository.

### Discover docs and templates

```bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
find "$REPO_ROOT/docs/bu-scc/jobs" -maxdepth 1 -type f -name "*.qsub" 2>/dev/null
```

Primary docs:
- `docs/bu-scc/setup/quickstart.md`
- `docs/bu-scc/runbooks/batch-notify.md`
- `docs/bu-scc/jobs/README.md`

### Storage and quota precheck


```bash
groups
pquota
```

Placement guidance:
- keep hand-edited code and configs in repository paths under project storage
- place large generated artifacts under project disk (`/project` or `/projectnb`) rather than home
- use node-local scratch only for temporary runtime intermediates

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

### BU SCC transfer and connectivity guardrails

- For large data movement, route through BU transferring-files guidance before suggesting transfer-node jobs.
- For interactive reconnectability, route through BU OnDemand guidance before suggesting qrsh-only flows.

### Infer with optional Notify

Prepare and inspect the runbook with `ops runbook init --workflow infer` and
`ops runbook plan`. Infer configurations define the model, records and feature
destination. Verify model requirements against the requested GPU resources.
Runbooks include Notify by default; use `--no-notify` for batch-only execution.
For notification setup and scheduler details, follow
`docs/operations/orchestration/runbooks.md` and
`docs/bu-scc/runbooks/batch-notify.md`.

Notify reads the destination dataset's `.events.log`. Preserve a separate cursor
and spool for each watched destination. Record both workload and watcher job IDs.


### Study-owned Infer workloads

Use the study's private workflow to select the exact workspace, views, model,
and runbook. Then use the generic fill command to plan, execute, or submit only
the missing sidecar work:

```bash
uv run ops runbook fill-infer --study-dir /path/to/study
```

Study workflows should use explicit `view_name` selectors. Completion planning
should reuse current-fingerprint sidecars, block missing sequence products, and
resume missing or stale vector/scalar sidecars through the normal Infer shard
ledger.
