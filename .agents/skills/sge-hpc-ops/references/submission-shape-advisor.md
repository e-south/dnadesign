## Submission Shape Advisor

Use this reference when deciding how to submit one or many jobs under current queue pressure.

### Purpose

- choose submission shape that respects shared queue behavior
- reduce scheduler churn from bursty independent submissions
- provide deterministic guidance to naive agents

### Queue fairness policy

- respect the queue and do not skip the line
- do not attempt queue-bypass behavior
- when busy, prefer efficient submission shape over submit bursts

### Standard command

```bash
$SKILL_DIR/scripts/sge-submit-shape-advisor.sh \
  --planned-submits <N> \
  --warn-over-running 3
```

Add `--requires-order` when jobs must run sequentially.
The supported `uv run ops runbook diagnostics submit-shape-advisor` command uses
the same machine-readable `advisor`, `reason`, and `recommended_action`
contract.

### Advisor outcomes

- `advisor=array`
  - use when many independent similar jobs are planned
  - rationale: lower scheduler overhead and clearer tracking
- `advisor=hold_jid`
  - use when jobs must run in order
  - rationale: explicit dependency chain without submit bursts
- `advisor=single`
  - use when one submit is planned; inspect `reason` and `recommended_action`
    to see whether high running-job pressure still requires confirmation
- `advisor=hold`
  - use when `Eqw` jobs are present; stop and triage before new submits

### Example flows

Independent fanout under high pressure:

```bash
$SKILL_DIR/scripts/sge-submit-shape-advisor.sh \
  --planned-submits 32 \
  --warn-over-running 3
```

Ordered pipeline under high pressure:

```bash
$SKILL_DIR/scripts/sge-submit-shape-advisor.sh \
  --planned-submits 8 \
  --requires-order \
  --warn-over-running 3
```
