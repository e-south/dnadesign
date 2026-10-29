## `baserender` for agents

Use this file with the repository-level `AGENTS.md`.

### Key paths
- README: `src/dnadesign/baserender/README.md`
- Example jobs: `src/dnadesign/baserender/docs/examples/`
- Demo workspaces: `src/dnadesign/baserender/workspaces/`
- Source: `src/dnadesign/baserender/src/`
  - Public API: `src/public/`
  - Job and style contracts: `src/config/`
  - Neutral records and errors: `src/core/`
  - Producer translation: `src/integrations/`
  - Selection and transform execution: `src/pipeline/`
  - Reusable visual grammars: `src/render/`
  - Output publication: `src/outputs/`
- Tests: `src/dnadesign/baserender/tests/`
- Generated outputs: workspace `outputs/`, job-local `results/`, or explicitly configured paths

### System dependency (video)
Video export requires **ffmpeg**. Don’t attempt to install system deps unless asked.

### Ownership boundary

- `core`, `pipeline`, `render`, and `outputs` stay producer-neutral.
- `integrations/<producer>/` translates an upstream contract into neutral
  records or supplies a built-in transform owned by that producer.
- Producer metrics, rankings, interpretation, and statistical plots stay with
  the producer.
- Built-in integrations are registered internally. Do not add entry-point
  discovery until an independently packaged integration needs it.
- New consumers use `dnadesign.baserender`, never
  `dnadesign.baserender.src.*`.

### Commands (copy/paste)
```bash
uv run baserender --help
uv run baserender catalog
uv run baserender catalog --json

uv run baserender job --help
uv run baserender style --help
uv run baserender workspace list --root src/dnadesign/baserender/workspaces
uv run baserender job validate --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces
```

Use the catalog before editing a job. It is the machine-readable source for
registered adapters, transforms, style profiles, renderers, and render
contracts.

### Tests

If you modify `baserender`, run the package suite before broader repo checks:

```bash
uv run pytest -q src/dnadesign/baserender/tests
uv run ruff check src/dnadesign/baserender pyproject.toml
uv run ruff format --check src/dnadesign/baserender
```
