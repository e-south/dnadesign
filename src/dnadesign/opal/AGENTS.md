## `opal` for agents

Supplement to repo-root `AGENTS.md` with `opal`-specific navigation + artifact rules.

### Key paths
- README: `src/dnadesign/opal/README.md`
- Docs index: `src/dnadesign/opal/docs/index.md`
- CLI manual: `src/dnadesign/opal/docs/reference/cli.md`
- Workflows: `src/dnadesign/opal/docs/workflows/`
- Campaign route index: `src/dnadesign/opal/campaigns/README.md`
- Campaigns: `src/dnadesign/opal/campaigns/<campaign>/`
  - `configs/campaign.yaml` (hand-edited input)
  - `state.json` (usually run-updated; treat as artifact)
  - `outputs/` (generated: events + per-round artifacts + ledgers)
- Source: `src/dnadesign/opal/src/`
- Tests: `src/dnadesign/opal/tests/`
- Notebooks: `src/dnadesign/opal/notebooks/`

### Generated vs hand-edited
- Hand-edited: `campaign.yaml`, code, docs
- Generated/run artifacts (do not hand-edit): `campaigns/**/outputs/**`
- Treat `campaigns/**/state.json` as a run artifact.
- Ask before committing changed `outputs/` or `state.json`.
- Checked-in campaign configs declare `ownership.owner_scope` as `opal_demo`
  or `study_campaign`. Configless historical run directories are not active
  campaign surfaces.
- A campaign owns one shared learning lifecycle. Target-specific objectives
  and selectors belong under named `selection_views`.

### Run and inspect a campaign

Follow the [campaign-round workflow](docs/workflows/campaign-round.md) for input
preparation, validation, label ingestion, selection and output verification.
For commands resolved from an existing config, run:

```bash
uv run opal guide -c /path/to/campaign/configs/campaign.yaml --format markdown
```

Use the [CLI reference](docs/reference/cli.md) for inspection and export options.
Start with the [isolated demo](docs/index.md#try-a-complete-selection-round) when
checking the workflow without changing an existing campaign.

### Tests

```bash
uv run pytest -q src/dnadesign/opal/tests
```
