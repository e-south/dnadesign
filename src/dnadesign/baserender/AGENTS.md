## `baserender` for agents

Supplement to repo-root `AGENTS.md` with baserender-specific layout + system deps.

### Key paths
- README: `src/dnadesign/baserender/README.md`
- Example jobs: `src/dnadesign/baserender/docs/examples/`
- Demo workspaces: `src/dnadesign/baserender/workspaces/`
- Source: `src/dnadesign/baserender/src/`
  - CLI implementation: `src/dnadesign/baserender/src/cli/`
  - Config/schema: `src/dnadesign/baserender/src/config/`
  - Renderers: `src/dnadesign/baserender/src/render/`
  - Output writers: `src/dnadesign/baserender/src/outputs/`
- Tests: `src/dnadesign/baserender/tests/`
- Generated outputs: workspace `outputs/`, job-local `results/`, or explicitly configured paths

### System dependency (video)
Video export requires **ffmpeg**. Don’t attempt to install system deps unless asked.

### Dataset contract (common)
- `sequence` (str)
- `densegen__used_tfbs_detail` (list[dict] annotations) for DenseGen-style overlays
- optional `id`

### Commands (copy/paste)
```bash
uv run baserender --help

uv run baserender job --help
uv run baserender style --help
uv run baserender workspace list --root src/dnadesign/baserender/workspaces
uv run baserender job validate --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces
```

### Tests

If you modify `baserender`, run the package suite before broader repo checks:

```bash
uv run pytest -q src/dnadesign/baserender/tests
uv run ruff check src/dnadesign/baserender pyproject.toml
uv run ruff format --check src/dnadesign/baserender
```
