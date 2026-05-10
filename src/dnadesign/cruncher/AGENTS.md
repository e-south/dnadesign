## `cruncher` for agents

Supplement to repo-root `AGENTS.md` with cruncher-specific layout + test guidance.

### Key paths
- README: `src/dnadesign/cruncher/README.md`
- Tracked study records: `docs/studies/README.md`
- Checked-in retron hairpin study: `docs/studies/retron_hairpin_design/`
- Repo-local retron hairpin study skill: `.agents/skills/retron-hairpin-study/SKILL.md`
- Default config: `src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml`
- Source: `src/dnadesign/cruncher/src/` (cli/, config/, core/, ingest/, io/, analysis/, artifacts/, viz/, integrations/, app/, store/, utils/)
- Results: per-workspace `out_dir` (demo default: `src/dnadesign/cruncher/workspaces/demo_basics_two_tf/outputs/`)
- Tests: `src/dnadesign/cruncher/tests/`
  - Slow tests: `src/dnadesign/cruncher/tests/slow/`

### Generated vs hand-edited
- Hand-edited: `workspaces/*/config.yaml`, code, test data under `tests/data/`
- Generated: `**/outputs/**`, `**/results/**` (batch folders, plots, traces)

### Commands (copy/paste)
First confirm the CLI surface:
```bash
uv run cruncher --help
```

Typical flow:

```bash
uv run cruncher parse   src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml
uv run cruncher sample  src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml
uv run cruncher analyze src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml
```

### Tests

```bash
uv run pytest -q src/dnadesign/cruncher/tests
```
