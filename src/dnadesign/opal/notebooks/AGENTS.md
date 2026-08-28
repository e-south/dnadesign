## Notebooks for agents

- Follow repo-root `AGENTS.md`.
- Canonical marimo rules: `docs/notebooks/marimo-reference.md`.
- Only edit code inside `@app.cell` bodies.
- Treat `src/dnadesign/opal/campaigns/**/outputs/**` as generated.
- If writing files, write to a clearly named local output dir and ask before committing.
- Generate operative campaign notebooks through `opal notebook generate`; do
  not fork generated cells into campaign-specific source modules.
- Multi-view notebooks use `Round | Selection view | Deliverable`. Shared
  model diagnostics appear once; target masks and selections resolve from the
  selected view.

### Run/edit
```bash
uv sync --locked

uv run opal notebook generate -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml \
  --round latest --force
uv run marimo check src/dnadesign/opal/campaigns/<campaign>/notebooks/*.py
uv run opal notebook run -c src/dnadesign/opal/campaigns/<campaign>/configs/campaign.yaml
```
