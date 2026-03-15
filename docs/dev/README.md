## Developer Documentation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-03

Use this index to find maintainer workflows, CI parity commands, and planning records.

### Start here

1. Read [repository docs index](../README.md) for the full docs map.
2. Review [PLANS](../../PLANS.md) before changing architecture or execution flow.
3. Use [architecture decisions index](../architecture/decisions/README.md) for approved decisions.

### Day-to-day tasks

1. Record implementation notes in [`journal.md`](journal.md).
2. Track structure and IA risks in [`monorepo-organization-audit.md`](monorepo-organization-audit.md).
3. Create or update proposals in [`plans/`](plans/).
4. Run docs checks before merging docs updates:
`uv run python -m dnadesign.devtools.docs_checks --repo-root .`
5. Run boundary checks when changing cross-tool imports:
`uv run python -m dnadesign.devtools.architecture_boundaries --repo-root .`
6. Run the repo-local BU SCC skill audit when changing `.agents/skills/sge-hpc-ops/`:
`bash .agents/skills/sge-hpc-ops/scripts/audit-sge-hpc-ops-skill.sh`

### CI and quality checks

- Core lane test expression: `-m "not fimo and not integration"`
- External integration test expression: `-m "fimo or integration"`
- Core lane local parity:
```bash
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
uv run pytest -q -m "not fimo and not integration"
uv run pytest -q -m "not fimo and not integration" --cov=src/dnadesign --cov-report=json:coverage-core.json
uv run python -m dnadesign.devtools.tool_coverage --coverage-json coverage-core.json --baseline-json .github/tool-coverage-baseline.json
uv run python -m dnadesign.devtools.coverage_summary --coverage-json coverage-core.json --baseline-json .github/tool-coverage-baseline.json --output-json quality-score-coverage-summary.json
uv run python -m dnadesign.devtools.quality_score --coverage-summary-json quality-score-coverage-summary.json --baseline-json .github/tool-coverage-baseline.json --core-lane-result success --external-integration-lane-result skipped --publish-lane-result skipped --output-json quality-score-inputs.json
```
- External integration local parity:
```bash
pixi install --locked
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
eval "$(PYTHONPATH=src python3 -m dnadesign.devtools.meme_env --repo-root . --print-shell-export)"
fimo --version
uv run pytest -q -m "fimo or integration" --junitxml external-integration-junit.xml
uv run python -m dnadesign.devtools.pytest_gate --junit-xml external-integration-junit.xml --lane-name external-integration --required-tools-csv "<tool1,tool2>"
```

### Planning and decisions

1. Proposal lifecycle and promotion rules: [PLANS](../../PLANS.md)
2. Execution plan indexes: [active plans](../exec-plans/active/README.md), [completed plans](../exec-plans/completed/README.md)
3. Decision records: [architecture decisions](../architecture/decisions/README.md)

### Naming and file layout

- Use kebab-case for markdown files.
- Prefix plan docs with `YYYY-MM-DD-`.
- Keep archived plans under `plans/archive/`.
