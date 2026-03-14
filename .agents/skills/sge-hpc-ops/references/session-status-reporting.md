## Session Status Reporting

Use this reference to produce a concise status report that both agent and user can reason about before new submissions.

### Purpose

- make current execution locus explicit
- report active SGE job pressure in one command
- apply a default warning gate when running job count exceeds 3

### Standard command

```bash
scripts/sge-session-status.sh --warn-over-running 3
```

Expected fields:
- host, user, working directory, execution-locus guess
- scheduler command availability (`qsub`, `qstat`, `qdel`)
- active job counts (`running_jobs`, `queued_jobs`, `hold_jobs`, `eqw_jobs`)
- threshold result (`threshold=3`, `threshold_exceeded=yes|no`)

### Active job snapshot command

Use this to report concrete job ids and states alongside aggregate counts:

```bash
scripts/sge-active-jobs.sh --max-jobs 12
```

### If more than 3 jobs are running

- emit a warning before proposing additional `qsub` commands
- recommend array conversion or dependency chaining (`-hold_jid`) first
- require explicit confirmation before additional parallel submissions

### Queue fairness guardrail

- respect the queue and do not skip the line
- avoid queue-bypass suggestions even when urgency is high
- pair status output with advisor output before multi-submit decisions

### Preferred operator UX command

Use this as the default end-user report when submit guidance is needed:

```bash
scripts/sge-operator-brief.sh --planned-submits <N> --warn-over-running 3
```

### Manual fallback commands

```bash
hostname
pwd
whoami
qstat -u "$USER" | sed -n '1,80p'
```

```bash
qstat -u "$USER" | awk '
  $1 ~ /^[0-9]+$/ {
    state=$5
    if (state ~ /r/) running++
    if (state ~ /q/) queued++
    if (state ~ /h/) hold++
    if (state ~ /Eqw/) eqw++
  }
  END { printf "running_jobs=%d queued_jobs=%d hold_jobs=%d eqw_jobs=%d\n", running, queued, hold, eqw }
'
```

### Deterministic fixture mode

Use fixture input for CI-style checks and offline validation:

```bash
scripts/sge-session-status.sh --qstat-file /tmp/qstat.fixture --warn-over-running 3
```

### Communication contract

When reporting status:
- include whether threshold was exceeded
- include exact warning rationale when threshold exceeded
- include next-step options (wait, chain with `-hold_jid`, or convert to array)
- include the rendered status card from `scripts/sge-status-card.sh` when interacting with end users
