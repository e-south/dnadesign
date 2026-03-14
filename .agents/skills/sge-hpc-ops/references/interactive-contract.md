## Interactive Contract

Use interactive mode only for short debugging and environment validation.

### Routing

- choose `interactive_ondemand_duo` when reconnectability or GUI usage is required
- choose `interactive_cli_qrsh` when the agent must execute directly in shell
- if intent is unclear, ask one routing question before submit
- for BU SCC routing context, reference `bu-scc-system-usage.md` entries for connect-scc and OnDemand

### Queue-safe rules

```bash
qstat -u "$USER"
```

- if a matching interactive request exists, reuse it
- if none exists, submit exactly one new interactive request
- use bounded wait (default 20s poll, 600s max)
- no automatic second submit

### CLI interactive behavior

- BU SCC interactive CLI methods are `qrsh` and `qsh`
- both default to 12h wallclock unless explicitly requested otherwise
- use `qrsh -now n` when queueing is desired instead of immediate-fail behavior
- login nodes are for light tasks; long CPU-bound work should move to batch due process-reaper enforcement

### OnDemand request flow

- treat OnDemand as user-authenticated flow requiring explicit confirmation
- require post-confirmation verification before claiming session-ready state
- OnDemand browser logout does not terminate running sessions; surface this before exit
- when policy-sensitive thresholds are cited (for example, the 5-job interactive limit), validate freshness in `source-evidence.md`
- for large-memory OnDemand sessions, require explicit extra qsub options based on current BU guidance

### OnDemand handoff contract

- when user states they are already in OnDemand, set `workflow_id=ondemand_session_handoff`
- do not re-issue session-request instructions in this path
- run lightweight in-session probes before task actions (`hostname`, `pwd`, `qstat -u "$USER"`)
- if requested task is long-running, route to batch submit and keep interactive context for diagnostics

### Guarded retry

Allow one retry only when all hold:
- explicit failure reason is recorded
- same request fingerprint is reused
- user confirms retry in the current turn
