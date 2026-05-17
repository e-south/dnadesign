## Operator Brief

Use this reference when the user asks for current HPC state and whether additional submits are safe.

### Purpose

- provide one concise report for agent and human
- reduce back-and-forth on "can we submit now?"
- keep submit decisions queue-fair and advisor-aligned

### Standard command

```bash
$SKILL_DIR/scripts/sge-operator-brief.sh \
  --planned-submits <N> \
  --warn-over-running 3
```

Add `--requires-order` for dependency-ordered pipelines.
The supported `uv run ops runbook diagnostics operator-brief` command uses the
same `submit_gate`, `advisor`, and `next_action` contract.

### Output fields

- `Submit Gate`: `ready`, `confirmation_required`, or `blocked`
- `Health`: `green`, `yellow`, or `red`
- `Execution Locus`: where commands are running
- `Running Jobs`, `Queued Jobs`, `Eqw Jobs`
- `Advisor`: `single`, `array`, `hold_jid`, or `hold`
- `Reason`, `Recommendation`, `Next Action`
- `Queue Policy`: `respect_queue`

Use `--json` for machine-readable fields including numeric `running_jobs`, `threshold`, and `planned_submits`.

When users ask for exact active job ids/states, pair operator brief with:

```bash
$SKILL_DIR/scripts/sge-active-jobs.sh --max-jobs 12
```

### Gate semantics

- `ready`: proceed with verify-before-submit and QA preflight
- `confirmation_required`: obtain explicit user confirmation before additional submits
- `blocked`: triage `Eqw` or failed jobs before any new submit

### Deterministic fixture mode

```bash
$SKILL_DIR/scripts/sge-operator-brief.sh \
  --qstat-file /tmp/qstat.fixture \
  --planned-submits 8 \
  --warn-over-running 3
```
