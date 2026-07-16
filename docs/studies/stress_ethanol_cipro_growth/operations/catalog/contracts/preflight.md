## Stress Ethanol Cipro Growth Preflight

**Type:** contract
**Plane:** data-plane
**Owner-boundary:** studies
**Entry artifact:** `docs/studies/stress_ethanol_cipro_growth/` plus the study-owned execution surfaces declared in `operations/ops.study.yaml`
**Exit artifact:** a read-only command-level readiness summary for this study
**Registry-id:** studies.stress-ethanol-cipro-growth.preflight
**Summary:** Run the stress_ethanol_cipro_growth preflight suite across its declared DenseGen, Construct, Infer, Notify, LatentDNA, and OPAL round-0 review surfaces.
**Execution-kind:** iterative
**Status-kind:** stress-ethanol-cipro-growth-preflight

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-15

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

The LatentDNA phase uses the study-owned semantic check
`latentdna.readiness.semantic` for primary X-selection readiness fields such as
`missing_source_datasets`, `missing_decision_deliverables`, and
`pending_deliverables`. Appendix-only RegulonDB/native review sources are
reported as appendix drift, not OPAL blockers.

The current main-path `--scope next` focus is `opal_round0_candidate_review`.
That readiness gate verifies the 27-label response-window promotion, validates
the campaign, loads the 18-sequence round-0 batch, and compares each of the
ethanol, ciprofloxacin, and AND selection artifacts with the run ledger. The
candidate-table materialization phase is complete and remains available in the
full-scope preflight as an upstream contract check.

These checks are read-only. Passing them routes to the campaign notebook and
review runbook; it does not authorize synthesis.

Use [routes](../../../routes/README.md) for the owner handoff after blockers are clear.
