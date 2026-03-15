## User Status Contract

Use this contract whenever an agent reports current HPC state to the user.

### Purpose

- make queue and risk posture legible in one short block
- standardize recommendation phrasing across workflows
- reduce ambiguous "can I submit now?" back-and-forth

### Command path

1. Collect machine status:

```bash
scripts/sge-session-status.sh --warn-over-running 3
```

2. Collect active job snapshot:

```bash
scripts/sge-active-jobs.sh --max-jobs 12
```

3. Render user-facing status card:

```bash
scripts/sge-status-card.sh --warn-over-running 3
```

4. Render consolidated operator brief for submit gating:

```bash
scripts/sge-operator-brief.sh --planned-submits <N> --warn-over-running 3
```

### Status card template

```text
HPC Status Card
- Health: <green|yellow|red>
- Execution Locus: <local_shell|scc_login_shell|ondemand_shell|ondemand_app_shell|unknown>
- Running Jobs: <int> (threshold <int>)
- Queued Jobs: <int>
- Eqw Jobs: <int>
- Reason: <short reason>
- Recommendation: <actionable next step>
```

### Health mapping

- `red`:
  - condition: `eqw_jobs > 0`
  - recommendation: triage `Eqw` before new submissions
- `yellow`:
  - condition: `running_jobs > threshold` and no `Eqw`
  - recommendation: confirm before additional submissions and prefer arrays or `-hold_jid`
- `green`:
  - condition: no `Eqw` and `running_jobs <= threshold`
  - recommendation: proceed with verify-before-submit gate

### Queue fairness note

- include a direct queue-fairness statement when proposing actions
- wording should state that the plan respects the queue and does not skip the line

### Reporting rules

- Always include the status card before proposing new submit commands.
- Include the operator brief when the user asks whether additional submits are safe.
- If health is `yellow` or `red`, include the explicit reason line and next action.
- Keep language direct; avoid generic reassurance.
