## Probe-First Contract

Run these probes before generating scheduler commands.

### Probe 0: execution locus and session context

```bash
hostname
pwd
whoami
echo "SSH_CONNECTION=${SSH_CONNECTION:-}"
command -v qsub qstat qdel >/dev/null && echo "SGE-like tools present" || true
```

Heuristic labels:
- `local_shell`: SGE tools absent and no SCC host markers
- `scc_login_shell`: SGE tools present and hostname/session indicates SCC login context
- `ondemand_shell` or `ondemand_app_shell`: user confirms OnDemand session, plus SCC shell probes pass
- `unknown`: insufficient evidence

### Probe A: scheduler identity and command availability

```bash
command -v qsub qstat qdel >/dev/null && echo "SGE-like tools present"
command -v qrsh >/dev/null && echo "qrsh available" || true
command -v qlogin >/dev/null && echo "qlogin available" || true
qsub -help 2>&1 | sed -n '1,120p'
qstat -help 2>&1 | sed -n '1,120p'
```

Stop condition:
- if `qsub` or `qstat` is missing for a scheduler-required workflow, stop and request scheduler context.

### Probe B: resource vocabulary (best effort)

```bash
qconf -spl 2>/dev/null || echo "[warn] PE list unavailable"
qconf -sc 2>/dev/null | sed -n '1,200p' || echo "[warn] complex list unavailable"
qconf -sql 2>/dev/null || echo "[warn] queue list unavailable"
```

### Probe C: account or project flag

```bash
qsub -help 2>&1 | grep -E ' -P | -A |project|account' || true
```

### Probe D: job activity and queue pressure

Prefer the helper command:

```bash
scripts/sge-session-status.sh --warn-over-running 3
```

Manual fallback if script is unavailable:

```bash
qstat -u "$USER" | awk '
  $1 ~ /^[0-9]+$/ {
    total++
    state=$5
    if (state ~ /r/) running++
    if (state ~ /q/) queued++
    if (state ~ /h/) hold++
    if (state ~ /Eqw/) eqw++
  }
  END {
    printf "total_jobs=%d running_jobs=%d queued_jobs=%d hold_jobs=%d eqw_jobs=%d\n", total, running, queued, hold, eqw
  }
'
```

If `running_jobs > 3` and the user asks for additional submits, emit warning and require explicit confirmation.

### Capability snapshot template

```text
Capability snapshot
- workflow_id: <densegen_batch_submit|densegen_batch_with_notify_slack|ondemand_session_request|ondemand_session_handoff|generic_sge_ops>
- execution_locus: <local_shell|scc_login_shell|ondemand_shell|ondemand_app_shell|unknown>
- session_handoff_state: <none|session_request_pending|session_ready>
- host: <hostname>
- scheduler_tools: qsub=<yes/no> qstat=<yes/no> qdel=<yes/no>
- interactive_cmd: <qrsh|qlogin|unknown>
- accounting_flag: <-P|-A|unknown>
- pe_known: <yes/no>
- walltime_key: <h_rt|unknown>
- memory_key: <mem_per_core|mem_free|h_vmem|unknown>
- gpu_keys: <list|unknown>
- transfer_key: <download|unknown>
- running_jobs: <int|unknown>
- queued_jobs: <int|unknown>
- eqw_jobs: <int|unknown>
- running_threshold: 3
- running_threshold_exceeded: <yes/no/unknown>
- unknowns: <comma-separated fields or none>
```
