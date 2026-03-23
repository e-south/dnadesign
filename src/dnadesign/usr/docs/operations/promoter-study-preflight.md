## Promoter Study Preflight

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** one checked-in promoter-study record plus its study-owned execution surfaces
**Exit artifact:** one read-only preflight summary across DenseGen, Construct, Infer, Notify, and batch-plan contracts

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-22

Use this contract after the cheaper
[Promoter Study Status Contract](promoter-study-status-contract.md) when you
need command-level answers to "what is ready, what is blocked, and why?" for
the real checked-in study.

Fastest active-study preflight:

```bash
uv run ops progress show usr.data-plane.promoter-study-preflight --json
```

If you need to pin a non-active study or you are invoking the command from
outside the repo checkout, add:

```bash
uv run ops progress show usr.data-plane.promoter-study-preflight \
  --repo-root <repo-root> \
  --study-dir docs/studies/promoter/<study-id> \
  --json
```

This route is still read-only. It does not submit jobs, mutate USR datasets, or
advance Notify cursors. It composes explicit command preflights from the
checked-in study record:

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
- Fail visibly when `campaign.yaml`, `datasets.yaml`, `status.md`, or declared
  execution surfaces are missing.
- Keep degraded state explicit:
  - missing datasets => `missing`
  - failed command preflights => `attention`
  - blocked GPU-only lanes remain visible; there is no hidden 20B -> 7B fallback
- Use the existing study-owned `pipeline.yaml` as the only source for real
  Construct, Infer, Notify, and runbook paths.

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
2. Run `usr.data-plane.promoter-study-preflight` when you need command-level
   blockers before the next DenseGen, Construct, Infer, or Notify step.
3. Use the returned `checks` list to decide whether the next concrete action is:
   - grow DenseGen again
   - materialize the merged anchor set
   - materialize Construct contexts
   - fix Notify secret/profile contracts
   - move Infer execution to a Hopper/H200-capable GPU node
