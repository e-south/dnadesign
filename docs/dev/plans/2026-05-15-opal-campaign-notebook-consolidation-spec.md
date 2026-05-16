## OPAL Campaign Notebook Consolidation Spec

**Date:** 2026-05-15
**Owner:** dnadesign maintainers
**Status:** active

### Intent

Make `opal notebook generate` the canonical campaign-specific viewing surface
for OPAL artifacts. The notebook should help users inspect campaign contracts,
round/run state, ledgers, records, labels, predictions, and plot deliverables
without becoming a competing control plane or a study-specific atlas.

### Best Outcome

A naive agent can enter through `promoter-study-status`, read the checked-in
study record, follow `docs/studies/<study-id>/routes.md` to OPAL, and use one
generated marimo notebook for campaign/round artifact visualization. OPAL CLI
commands remain canonical for state and validation; the notebook composes their
concepts into an ergonomic read-only view.

### Non-Goals

- Do not add OPAL command sprawl when existing commands can express the flow.
- Do not embed OPAL walkthroughs directly in `promoter-study-status`.
- Do not add BaseRender-specific or DenseGen-specific rendering logic to OPAL.
- Do not make LatentDNA projection or atlas columns OPAL readiness gates.
- Do not maintain a checked-in active notebook as a second user-facing route
  that competes with generated campaign notebooks.

### Architecture Decisions

1. `opal notebook generate` is the campaign-specific artifact viewer.
2. Checked-in notebook code is a maintainer fixture/template reference, not a
   parallel operator route.
3. OPAL CLI remains canonical for `validate`, `status`, `runs`,
   `record-show`, `verify-outputs`, `plot`, and notebook generation/running.
4. Shared helper modules return plain data, strings, or typed records; marimo
   cells stay focused on UI composition.
5. Study and ops routes point to OPAL through `routes.md` and OPAL docs, while
   `promoter-study-status` remains the record-plane router.

### Notebook Contract

Generated notebooks should use `mo.accordion` progressive disclosure with these
sections:

- `Campaign contract`
- `Round and run`
- `Ledger readiness`
- `Records and active record`
- `Labels and predictions`
- `Plot deliverables`
- `Optional context boundaries`

The notebook should show plot deliverables from `outputs/plots`, configured
plot entries, missing expected plot outputs, and enough file context for a user
to decide whether to regenerate plots with `opal plot`.

### Study/Ops Routing Contract

For promoter studies, `promoter-study-status` answers the record-backed status
question, then routes OPAL follow-up through `docs/studies/<study-id>/routes.md`.
The study route should include:

- candidate feature table id and role
- campaign config path(s)
- campaign workdir or enough config context to resolve it
- minimal OPAL CLI handoff commands
- `opal notebook generate` and `opal notebook run` as the campaign notebook path

### Progress

- [x] Archived legacy `prom60_eda` under `src/dnadesign/opal/archived/`.
- [x] Added `analysis/campaign_progress.py` helper boundary for records,
  ledger status, previews, and CLI handoff text.
- [x] Added regression tests that forbid BaseRender, `densegen__visual`, and
  UMAP readiness coupling in OPAL notebook surfaces.
- [x] Updated generated notebook template to use accordion sections.
- [x] Updated stress promoter route to advertise generated OPAL notebooks as
  the campaign-specific artifact viewer.
- [x] Changed `opal notebook generate` to work for valid pre-run campaigns;
  generated notebooks now show missing runs, labels, predictions, and plots as
  explicit degraded states instead of blocking generation on ledger artifacts.
- [x] Added promoter-study-status audit guards for the OPAL route boundary:
  status/plot/notebook questions route through `routes.md`, not the skill body.
- [ ] Dogfood generated notebooks against a demo campaign with plots present.
- [ ] Add adversarial CLI tests for missing plots and plot-output metadata
  where coverage is still thin.
- [ ] Decide whether the checked-in `campaign_progress.py` remains a fixture or
  is replaced by generator-backed tests only.

### Validation Checklist

```bash
uv run marimo check src/dnadesign/opal/notebooks/campaign_progress.py
DNADESIGN_HEADLESS=1 uv run pytest -q \
  src/dnadesign/opal/tests/notebooks \
  src/dnadesign/opal/tests/analysis \
  src/dnadesign/opal/tests/cli/test_cli_notebook_generate.py
uv run pytest -q src/dnadesign/studies/tests/test_stress_ethanol_cipro_opal_batch0.py
bash .agents/skills/promoter-study-status/scripts/audit-promoter-study-status-skill.sh
uv run ruff check src/dnadesign/opal src/dnadesign/studies .agents/skills/promoter-study-status
uv run ruff format --check src/dnadesign/opal src/dnadesign/studies
uv run python -m dnadesign.devtools.docs.checks --repo-root .
git diff --check
```

### Drift Guards

- Generated notebooks must not import `dnadesign.baserender`.
- Generated notebooks must not reference `densegen__visual`.
- Projection fields such as LatentDNA UMAP columns are optional context only.
- New study OPAL routing should update `routes.md`, not
  `promoter-study-status` top-level workflow text.
- New plot viewer behavior should extend OPAL plot artifact contracts before
  adding file-name-specific notebook logic.
