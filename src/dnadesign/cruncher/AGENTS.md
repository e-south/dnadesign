## `cruncher` for agents

Supplement to repo-root `AGENTS.md` with cruncher-specific layout + test guidance.

### Key paths
- README: `src/dnadesign/cruncher/README.md`
- Tracked study records: `docs/studies/README.md`
- Checked-in retron hairpin study: `docs/studies/retron_hairpin_design/`
- Repo-local retron hairpin study skill: `.agents/skills/retron-hairpin-study/SKILL.md`
- Default config: `src/dnadesign/cruncher/workspaces/demo_pairwise/configs/config.yaml`
- Source: `src/dnadesign/cruncher/src/` (cli/, config/, core/, ingest/, io/, analysis/, artifacts/, viz/, integrations/, app/, store/, utils/)
- Results: per-workspace `out_dir`; treat generated output directories as run artifacts.
- Tests: `src/dnadesign/cruncher/tests/`
  - Slow tests live under the tests tree when present.

### Generated vs hand-edited
- Hand-edited: `workspaces/*/configs/config.yaml`, code, test fixtures under `src/dnadesign/cruncher/tests/`
- Generated: `**/outputs/**`, `**/results/**` (batch folders, plots, traces)

### Commands (copy/paste)
First confirm the CLI surface:
```bash
uv run cruncher --help
```

Typical flow:

```bash
uv run cruncher parse --force-overwrite -c src/dnadesign/cruncher/workspaces/demo_pairwise/configs/config.yaml
uv run cruncher sample --force-overwrite -c src/dnadesign/cruncher/workspaces/demo_pairwise/configs/config.yaml
uv run cruncher analyze --summary -c src/dnadesign/cruncher/workspaces/demo_pairwise/configs/config.yaml
```

### Tests

```bash
uv run pytest -q src/dnadesign/cruncher/tests
```
