## Promoter Study Preflight

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** one checked-in promoter-study directory plus study-owned execution surfaces
**Exit artifact:** one read-only command-level preflight summary for the active study
**Registry-id:** usr.data-plane.promoter-study-preflight
**Summary:** Run the active promoter-study preflight suite across DenseGen, Construct, Infer, Notify, and batch-plan contracts without mutating data or submitting jobs.
**Execution-kind:** iterative
**Status-kind:** promoter-study-preflight

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-25

Use this contract after the cheaper
[Promoter Study Status Contract](promoter-study-status-contract.md) when you
need command-level answers to "what is ready, what is blocked, and why?" for
the real checked-in study.
This remains an observation-plane route: it composes read-only preflight checks
from the checked-in study record and still defers actual submit/execute work to
the control-plane `ops runbook` commands.

Fastest active-study preflight:

```bash
# Read the active study's command-level preflight summary as JSON.
uv run ops progress show usr.data-plane.promoter-study-preflight --json
```

Fastest next-phase preflight when you want the immediate actionable lane without
later-lane blocker noise:

```bash
# Focus the summary on the next actionable study phase and defer later-lane blockers.
uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json
```

If you need to pin a non-active study or you are invoking the command from
outside the repo checkout, add:

```bash
# Pin a specific checked-in study directory and emit the same preflight summary.
uv run ops progress show usr.data-plane.promoter-study-preflight \
  --repo-root <repo-root> \
  --study-dir docs/studies/promoter/<study-id> \
  --json
```

This route is still read-only. It does not submit jobs, mutate USR datasets, or
advance Notify cursors. It composes explicit command preflights from the
checked-in study record and uses `ops.study.yaml` to decide which study phases
belong to the next actionable scope versus the full study surface:

- DenseGen config probe from the study's batch runbook
- DenseGen batch `ops runbook plan`
- Construct workspace doctor
- Construct runtime validation when the merged-anchor dataset exists
- Infer config validation for the study-owned configs
- Infer Notify profile readiness, including exact `notify setup slack` commands
  when the lane profile has not been materialized yet
- Infer dry-run when the required study-owned USR datasets exist
- Notify event-path resolution for the same study configs
- Notify profile doctor for materialized Infer profiles, so missing TLS or
  webhook contracts fail visibly before live Slack delivery
- Infer batch `ops runbook plan` for the lane-specific Notify-backed presets

### Contract rules

- Start with `docs/studies/promoter/index.yaml` or pass `--study-dir`
  explicitly. Do not scan the repo for a best guess.
- Resolve relative study paths against the repo root, not the shell cwd.
- Fail visibly when `campaign.yaml`, `datasets.yaml`, `status.md`,
  `ops.study.yaml`, or declared execution surfaces are missing.
- Keep degraded state explicit:
  - missing datasets => `missing`
  - failed command preflights => `attention`
  - blocked GPU-only lanes remain visible; there is no hidden 20B -> 7B fallback
- Use `ops.study.yaml` as the OPS-facing source of phase order, snapshot scope,
  and preflight phase-target grouping.
- Use the existing study-owned `pipeline.yaml` as the only source for real
  Construct, Infer, and runbook paths plus any minimal runtime mappings the
  study still needs.
- Derive Infer Notify profile paths from the checked-in Infer lane configs
  rather than duplicating those profile paths in `pipeline.yaml`.

### What this route is for

- naive-agent status refresh before deciding the next command
- read-only readiness checks on login or CPU-only nodes
- explicit blocker reporting for missing USR datasets, Notify secret contracts,
  TLS/profile contracts, solver backends, and GPU-only lanes

### What this route is not for

- replacing tool-local runbooks
- mutating datasets or materializing new study artifacts
- hiding missing prerequisites behind fallback behavior

### Typical use

1. Run `usr.data-plane.promoter-study-status` for the cheap study snapshot.
   That summary is repo-scoped and does not elevate solely because the local
   host lacks a GPU.
2. Run `usr.data-plane.promoter-study-preflight` when you need command-level
   blockers before the next DenseGen, Construct, Infer, or Notify step.
3. Use the returned `checks` list to decide whether the next concrete action is:
   - grow DenseGen again
   - materialize the merged anchor set
   - materialize Construct contexts
   - fix Notify secret/profile contracts
   - move Infer execution to a Hopper/H200-capable GPU node
