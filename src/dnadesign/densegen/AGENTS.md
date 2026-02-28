## `densegen` for densegen-maintainers

Use this file as a short map. Keep detailed operating and architecture material in linked docs.

### Entry points
- Package overview: `src/dnadesign/densegen/README.md`
- Docs index by workflow: `src/dnadesign/densegen/docs/README.md`
- Workspace catalog: `src/dnadesign/densegen/workspaces/catalog.md`
- Workspace directory policy: `src/dnadesign/densegen/workspaces/README.md`

### Pick a path
- Run a packaged workspace: open the workspace directory and execute `./runbook.sh`.
- Debug run behavior: use `docs/concepts/quick-checklist.md`, then `docs/concepts/pipeline-lifecycle.md`.
- Check CLI behavior: `docs/reference/cli.md`.
- Check config fields: `docs/reference/config.md`.
- Check outputs and contracts: `docs/reference/outputs.md`.
- Run on clusters: `docs/howto/hpc.md` and `docs/howto/bu-scc.md`.

### Workspace conventions
- Packaged templates: `demo_tfbs_baseline`, `demo_sampling_baseline`, `study_constitutive_sigma_panel`, `study_stress_ethanol_cipro`.
- Generated outputs stay under each workspace `outputs/` directory.
- For USR sinks, stage workspaces with `dense workspace init --output-mode usr|both`.

### Core commands
```bash
uv run dense --help
uv run dense workspace where --format json
uv run dense inspect run --root src/dnadesign/densegen/workspaces
```

### Validation checks after changes
```bash
uv run pytest -q src/dnadesign/densegen/tests/docs/test_densegen_docs_ia_contracts.py
uv run pytest -q src/dnadesign/densegen/tests/docs/test_workspace_runbook_contracts.py
uv run python -m dnadesign.devtools.docs_checks
```
