## Developer Documentation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-27

Use this index to find maintainer workflows, checks, and planning records.

### Start here

1. Review [PLANS](../../PLANS.md) before changing architecture or execution flow.
2. Use [architecture decisions index](../architecture/decisions/README.md) for approved decisions.
3. Use [CI and quality checks](#ci-and-quality-checks) before merging maintainer changes.
4. Use the repo-local gate here rather than `./scripts/agent-verify`. That script belongs to the external agent-hub repo and is not present in `dnadesign`.

### Quick checks by change type

| If you changed | Run this first |
| --- | --- |
| docs, READMEs, or runbooks | `uv run python -m dnadesign.devtools.docs.checks --repo-root .` |
| cross-tool imports or ownership boundaries | `uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .` |
| OPS CLI entrypoints, parsing, or error rendering | `uv run pytest -q src/dnadesign/ops/tests/test_cli_failure_contract.py` |
| OPS native gate stderr or audit-fidelity behavior | `uv run pytest -q src/dnadesign/ops/tests/test_sge_gates.py -k run_native_gate_command_surfaces_failure_text_to_stderr` |
| OPS orchestration state or active-job discovery | `uv run pytest -q src/dnadesign/ops/tests/test_runbook_orchestrator.py -k "explicit_job_identity or discover_active_job_ids"` |
| OPS status aggregation semantics | `uv run pytest -q src/dnadesign/ops/tests/test_state_semantics.py` |
| code in one tool | `uv run pytest -q <tool test path>` and then broaden to the repo-level checks when the slice is stable |

### Day-to-day tasks

1. Record implementation notes in [`journal.md`](journal.md).
2. Track structure and IA risks in [`audits/monorepo-organization.md`](audits/monorepo-organization.md).
3. Create or update proposals in [`plans/`](plans/).
4. Run docs checks before merging docs updates:
`uv run python -m dnadesign.devtools.docs.checks --repo-root .`
5. Run boundary checks when changing cross-tool imports:
`uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .`
6. Run the repo-local skill audits when changing `.agents/skills/`:
`bash .agents/skills/stress-ethanol-cipro-growth-status/scripts/audit-stress-ethanol-cipro-growth-status-skill.sh`
`bash .agents/skills/retron-hairpin-study/scripts/audit-retron-hairpin-study-skill.sh`
`bash .agents/skills/notify-ops/scripts/audit-notify-ops-skill.sh`
`bash .agents/skills/bu-scc-usr-sync/scripts/audit-bu-scc-usr-sync-skill.sh`
`bash .agents/skills/sge-hpc-ops/scripts/audit-sge-hpc-ops-skill.sh`
7. Run the subtree routing contract checks when changing critical `AGENTS.md` surfaces or repo-local workflow skills:
`uv run pytest -q src/dnadesign/notify/tests/docs/test_progressive_disclosure_contracts.py`
`uv run pytest -q src/dnadesign/usr/tests/test_usr_docs_contract.py`
8. Run the OPS subprocess failure suite when changing console wiring or CLI contract text:
`uv run pytest -q src/dnadesign/ops/tests/test_cli_failure_contract.py`
9. Run the native gate stderr regression when changing `dnadesign.ops.orchestrator.gates` or any audit-fidelity path that executes those commands:
`uv run pytest -q src/dnadesign/ops/tests/test_sge_gates.py -k run_native_gate_command_surfaces_failure_text_to_stderr`
`uv run pytest -q src/dnadesign/ops/tests/test_runbook_orchestrator.py -k captures_gate_stderr_for_nonzero_native_gate_command`
10. Run focused OPS contract suites when changing state aggregation, preflight, or active-job identity:
`uv run pytest -q src/dnadesign/ops/tests/test_state_semantics.py`
`uv run pytest -q src/dnadesign/ops/tests/test_runbook_orchestrator.py -k "explicit_job_identity or discover_active_job_ids"`

### Repo-local maintainer gate

Run this from the repo root for tracked changes in `dnadesign`:

```bash
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run python -m dnadesign.devtools.docs.checks --repo-root .
```

### CI and quality checks

- Core lane test expression: `-m "not fimo and not integration"`
- External integration test expression: `-m "fimo or integration"`
- Core lane local parity:
```bash
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
uv run pytest -q -m "not fimo and not integration"
uv run pytest -q -m "not fimo and not integration" --cov=src/dnadesign --cov-report=json:coverage-core.json
uv run python -m dnadesign.devtools.quality.tool_coverage --coverage-json coverage-core.json --baseline-json .github/tool-coverage-baseline.json
uv run python -m dnadesign.devtools.quality.coverage_summary --coverage-json coverage-core.json --baseline-json .github/tool-coverage-baseline.json --output-json quality-score-coverage-summary.json
uv run python -m dnadesign.devtools.quality.score --coverage-summary-json quality-score-coverage-summary.json --baseline-json .github/tool-coverage-baseline.json --core-lane-result success --external-integration-lane-result skipped --publish-lane-result skipped --output-json quality-score-inputs.json
```
- External integration local parity:
```bash
pixi install --locked
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
eval "$(PYTHONPATH=src python3 -m dnadesign.devtools.runtime.meme_env --repo-root . --print-shell-export)"
fimo --version
uv run pytest -q -m "fimo or integration" --junitxml external-integration-junit.xml
uv run python -m dnadesign.devtools.runtime.pytest_gate --junit-xml external-integration-junit.xml --lane-name external-integration --required-tools-csv "<tool1,tool2>"
```

### Planning and decisions

1. Proposal lifecycle and promotion rules: [PLANS](../../PLANS.md)
2. Execution plan indexes: [active plans](../exec-plans/active/README.md), [completed plans](../exec-plans/completed/README.md)
3. Decision records: [architecture decisions](../architecture/decisions/README.md)
4. Sequence-view, reference-product, reverse-complement context, and Infer-completion hardening proposal:
   [2026-04-28 sequence-view ontology and Infer completion spec](plans/cross-tool/sequence-view/2026-04-28-ontology-and-infer-completion-hardening.md)
5. Generic linear ssDNA composition proposal for Construct, folding QA, BaseRender handoff, and Retron dogfooding:
   [2026-05-13 generic linear ssDNA composition spec](plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md)
6. Completed implementation record:
   [generic linear ssDNA composition execution plan](../exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md)
7. Active follow-up checklist:
   [linear ssDNA composition hardening follow-ups](../exec-plans/active/2026-05-14-linear-ssdna-composition-hardening-followups.md)
8. Active OPAL notebook consolidation spec:
   [OPAL campaign notebook consolidation](plans/tools/opal/2026-05-15-campaign-notebook-consolidation.md)

### Naming and file layout

- Use kebab-case for markdown files.
- Prefix plan docs with `YYYY-MM-DD-`.
- Keep current design proposals in semantic lanes under `plans/tools/` or
  `plans/cross-tool/`; keep archived plans under `plans/archive/`.
