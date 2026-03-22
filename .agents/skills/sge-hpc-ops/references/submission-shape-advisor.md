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
scripts/sge-submit-shape-advisor.sh \
  --planned-submits <N> \
  --warn-over-running 3
```

Add `--requires-order` when jobs must run sequentially.

### Advisor outcomes

- `advisor=array`
  - use when many independent similar jobs are planned
  - rationale: lower scheduler overhead and clearer tracking
- `advisor=hold_jid`
  - use when jobs must run in order
  - rationale: explicit dependency chain without submit bursts
- `advisor=single_submit`
  - use when one submit is planned and pressure is acceptable
- `advisor=confirm_then_submit`
  - use when running pressure is high but only one additional submit is requested
- `advisor=triage_first`
  - use when `Eqw` jobs are present; fix failures before new submits

### Example flows

Independent fanout under high pressure:

```bash
scripts/sge-submit-shape-advisor.sh \
  --planned-submits 32 \
  --warn-over-running 3
```

Ordered pipeline under high pressure:

```bash
scripts/sge-submit-shape-advisor.sh \
  --planned-submits 8 \
  --requires-order \
  --warn-over-running 3
```
