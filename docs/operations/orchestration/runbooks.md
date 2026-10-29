## Orchestration runbooks

**Type:** runbook
**Plane:** control-plane
**Owner-boundary:** ops
**Entry artifact:** workspace-scoped ops runbook YAML plus scheduler and optional notify intent
**Exit artifact:** audit JSON, rendered command plan, and optional submit or watcher contract
**Registry-id:** ops.control-plane.orchestration
**Summary:** Deterministic control-plane runbook contract for Infer batch submit flows with optional Notify chaining.
**Execution-kind:** executable
**Status-kind:** ops-audit-json

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-29

This contract defines machine-readable runbooks for cross-tool BU SCC control-plane orchestration.
It does not own durable SeqLedger-backed data-plane workflows; return to the root docs router or SeqLedger operations docs when the next procedure is about shared datasets rather than scheduler sequencing.

### Prepare a runbook

OPS plans and submits Infer batches to an SGE scheduler. A runbook binds one
workspace configuration to resource requests, log locations and optional Notify
watching. Planning does not submit a job.

```bash
# Create an Infer runbook beside the workspace's logs; substitute your scheduler project.
uv run ops runbook init \
  --workflow infer \
  --workspace-root /path/to/workspace \
  --project <project> \
  --runbook /path/to/workspace/outputs/logs/ops/runbooks/infer.yaml
# Inspect the resolved commands and resource requests before execution.
uv run ops runbook plan --runbook /path/to/workspace/outputs/logs/ops/runbooks/infer.yaml
```

Use `ops runbook init --help` for resource overrides and `--no-notify` for a
batch-only runbook. `ops runbook presets` lists optional scheduler defaults;
`--preset bu-scc-dunlop` selects the BU SCC project and job-template defaults.
The workload still belongs to Infer, and its configuration declares the model,
input records and output destination.

### Paths and resources

Keep the runbook at `<workspace-root>/outputs/logs/ops/runbooks/<id>.yaml`.
Relative paths resolve from the runbook's directory. The Infer configuration must
be a workspace-local YAML file whose name starts with `config`. Logs live under
`<workspace-root>/outputs/logs/ops/sge/<id>`; notification state lives under
`<workspace-root>/outputs/notify/infer`.

The supported workflow IDs are `infer_batch_submit` and
`infer_batch_with_notify`. Both require a GPU count and capability. The notify
workflow also requires an Infer notification source and the `infer` or `generic`
policy. The generated YAML contains the resource and path fields to review.

### Plan, execute and inspect

```bash
# Discover jobs already associated with this runbook.
uv run ops runbook active-jobs --runbook /path/to/workspace/outputs/logs/ops/runbooks/infer.yaml
# Review automatic fresh/resume selection and all preflight commands.
uv run ops runbook plan --runbook /path/to/workspace/outputs/logs/ops/runbooks/infer.yaml --mode auto
# Execute the reviewed plan and retain its audit in the workspace.
uv run ops runbook execute --runbook /path/to/workspace/outputs/logs/ops/runbooks/infer.yaml \
  --audit-json /path/to/workspace/outputs/logs/ops/audit/infer.json
```

Preflight checks validate required files, output locations, notification setup
and scheduler constraints before submission. Audits record command outcomes and
job IDs; supplied environment values are redacted. An execution failure retains
completed-command evidence. Inspect that audit and scheduler state before retrying.

Automatic mode checks the declared Infer destination for existing overlays.
Fresh execution with existing results requires an explicit reset acknowledgement;
use `--mode fresh --allow-fresh-reset` only when replacing prior outputs is intended.
Use `ops runbook plan --help` for mode and active-job options. Planning should be
reviewed again after changing the configuration or resource requests.

Execution uses `--command-timeout-seconds` (default `300`) for each command.
Only workspace-scoped audit paths are accepted. Active-job discovery normally
remains enabled; `--no-discover-active-jobs` requires an explicit
`--allow-unknown-active-jobs` acknowledgement.

Log retention uses `logging.retention.keep_last` and
`logging.retention.max_age_days`. GPU preflight checks `resources.gpu_memory_gib`
and the model's parallelism settings. Sequence-view Infer jobs retain their
per-feature completion rules rather than treating an unrelated overlay as a
completed feature bundle.

### Fill remaining Infer lanes

`ops runbook fill-infer` plans the remaining workspace-local Infer configurations
against active jobs and available lane capacity. For study-owned runs, use
`uv run ops runbook fill-infer --study-dir /path/to/study`; explicit `--runbook`
arguments select individual lanes. Start with its `--help` and the
plan output; execution remains explicit. Use distinct configuration files and
notification state for independent lanes.

### Related documentation

- [Infer](../../../src/dnadesign/infer/) owns model execution and feature output.
- [Notify](../../notify/) describes event sources and delivery configuration.
- [SGE job templates](../../bu-scc/jobs/) provide scheduler examples.
- [Operations](../README.md) covers status and diagnostic commands.
