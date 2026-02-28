## `densegen` for densegen-maintainers

Use this file as a short map. Keep detailed operating and architecture material in linked docs.

### Entry points
- [Package overview](README.md)
- [Docs index by workflow](docs/README.md)
- [Workspace catalog](workspaces/catalog.md)
- [Workspace directory policy](workspaces/README.md)

### Pick a path
- Run a packaged workspace: open the workspace directory and execute `./runbook.sh`.
- Debug run behavior: start with [Quick checklist](docs/concepts/quick-checklist.md), then [Pipeline lifecycle](docs/concepts/pipeline-lifecycle.md).
- Check CLI behavior: [CLI reference](docs/reference/cli.md).
- Check config fields: [Config reference](docs/reference/config.md).
- Check outputs and contracts: [Outputs reference](docs/reference/outputs.md).
- Run on clusters: [HPC runbook](docs/howto/hpc.md) and [BU SCC guide](docs/howto/bu-scc.md).

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
