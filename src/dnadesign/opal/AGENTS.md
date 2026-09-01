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

### Commands

Check available commands and flags:
```bash
uv run opal --help
```

Initialize and validate a campaign:
```bash
uv run opal init     -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml
uv run opal validate -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml
```

Ingest observed labels for a round and execute training/scoring/selection:
```bash
uv run opal ingest-y  -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml \
  --round 0 --csv <labels.xlsx> --apply
uv run opal run       -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml \
  --round 0

uv run opal status      -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml
uv run opal record-show -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml --id <id>
uv run opal explain     -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml --round 1
uv run opal plot        -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml
uv run opal predict     -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml --round latest
uv run opal selection-set show -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml \
  --view <selection-view-id> --round latest
uv run opal selection-batch show -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml \
  --round latest
```

### Tests

```bash
uv run pytest -q src/dnadesign/opal/tests
```
