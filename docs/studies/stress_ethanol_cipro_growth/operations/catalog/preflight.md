## Stress Ethanol Cipro Growth Preflight

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** studies
**Entry artifact:** `docs/studies/stress_ethanol_cipro_growth/` plus the study-owned execution surfaces declared in `operations/ops.study.yaml`
**Exit artifact:** a read-only command-level readiness summary for this study
**Registry-id:** studies.stress-ethanol-cipro-growth.preflight
**Summary:** Run the stress_ethanol_cipro_growth preflight suite across its declared DenseGen, Construct, Infer, Notify, and batch-plan surfaces.
**Execution-kind:** iterative
**Status-kind:** stress-ethanol-cipro-growth-preflight

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-17

Use this after [status](status.md) when the question is blocker or
next-run readiness for `stress_ethanol_cipro_growth`.

### Direct Commands

```bash
uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --json --command-timeout-seconds 30
```

```bash
uv run ops progress show studies.stress-ethanol-cipro-growth.preflight \
  --repo-root <repo-root> \
  --study-dir docs/studies/stress_ethanol_cipro_growth \
  --scope next \
  --json \
  --command-timeout-seconds 30
```

### Contract

The preflight provider composes read-only checks from this study's
`operations/ops.study.yaml`. It does not submit jobs, mutate USR datasets, advance Notify
cursors, or infer a hidden readiness graph from generic runbooks.

`--scope next` uses the study-owned lifecycle contract to focus the next
actionable phase and defer later-lane blockers. `--scope full` reports the full
declared suite.

Use [routes](../../routes/README.md) for the owner handoff after blockers are clear.
