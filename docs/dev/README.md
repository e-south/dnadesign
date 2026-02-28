## Developer Documentation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-19

This directory contains developer-facing planning and maintenance notes.

### Contents

- `journal.md`: implementation journal and decision history.
- `monorepo-organization-audit.md`: information architecture and layout audit notes.
- `plans/`: dated design proposals and architecture plans.
- `../../PLANS.md`: planning lifecycle map and pointers to execution plans/ADRs.
- `../../docs/architecture/decisions/README.md`: ADR conventions and numbering policy.

### Naming convention

- Use kebab-case for markdown files.
- Prefix plan documents with `YYYY-MM-DD-`.
- Keep archived plans under `plans/archive/`.

### CI test markers

- `fimo`: tests requiring MEME/FIMO tooling.
- `integration`: cross-component integration tests.
- External integration lane PR triggering is tool-agnostic and derived from tools that contain tests marked `fimo` or `integration`.
- External integration marker discovery inspects `tests/test_*.py` and `tests/conftest.py` for `pytestmark` assignments and mark decorators (`pytest.mark.<name>`, aliased `pytest`, and `from pytest import mark as ...`).
- MEME/FIMO path resolution for the external integration lane is centralized in `dnadesign.devtools.meme_env` (fails fast if `.pixi` FIMO is missing).
- CI changed-file collection is centralized in `dnadesign.devtools.ci_changed_files`.
- `ci_changed_files` diffs against an existing `<remote>/<base-ref>` tracking ref when present and fetches only when that ref is missing.
- `ci_changes` enforces a strict tool inventory contract: `.github/tool-coverage-baseline.json` must match repository tool directories under `src/dnadesign/` (excluding `devtools`, `archived`, and `prototypes`).
- Affected tool test-directory resolution is centralized in `dnadesign.devtools.ci_test_targets`.
- Architecture boundary enforcement is centralized in `dnadesign.devtools.architecture_boundaries` and runs in the core lane.
- Architecture boundary checks target production tool packages and exclude `devtools`, `archived`, and `prototypes`.
- Docs checks enforce root SOR metadata, knowledge-base/operator index metadata (`Owner`, `Last verified`), and exec-plan metadata/link traceability through `dnadesign.devtools.docs_checks`.
- Docs checks also enforce metadata (`Owner`, `Last verified`) for selected operational runbooks: installation/dependencies/notebooks/marimo and BU SCC + Notify event operator runbooks.
- Docs checks enforce the root README tool catalog contract: table rows must match repo tool inventory and each tool link must target `src/dnadesign/<tool>`.
- Docs checks enforce the Codecov component contract in `codecov.yml`: component ids must match tool inventory, each component must include `src/dnadesign/<tool>/**`, and `component_management.default_rules.statuses` must include a project status with strict failure semantics.
- Exec-plan checks also enforce required living-plan sections (`Purpose / Big Picture`, `Progress`, `Decision Log`, and related sections) for files under `docs/exec-plans/active/` and `docs/exec-plans/completed/`.
- Exec-plan checks enforce that checklist items appear under `Progress` only.
- Core lane uses `-m "not fimo and not integration"` and scopes PR runs to affected tool `tests/` directories only.
- Core lane installs `ffmpeg` so baserender rendering tests run in CI instead of being environment-skipped.
- External integration lane uses `-m "fimo or integration"` and scopes PR runs to affected tool `tests/` directories; non-PR, global-risk PR changes, and shared-package changes under `src/dnadesign/` outside known tools run the full external integration marker set.
- External integration lane writes JUnit XML and enforces a non-skipped execution gate via `dnadesign.devtools.pytest_gate`; all-skipped external integration runs fail, and each in-scope external integration tool must execute at least one non-skipped test.
- Core and external integration lanes upload XML coverage reports to Codecov using GitHub OIDC (`use_oidc: true`) with strict upload failure handling (`fail_ci_if_error: true`).
- Core-lane pre-commit runs on the PR diff (`--from-ref` / `--to-ref`); non-PR events run full `--all-files`.
- When a scoped PR has no affected tools, CI runs a real smoke-test subset without coverage upload and skips per-tool coverage/quality-score generation (`run_coverage_gate=false`).
- Coverage baselines are core-lane floors and must match this marker expression.
- Coverage dashboard semantics are explicit:
  - `actual`: measured core-lane line coverage for a tool.
  - `baseline`: required minimum from `.github/tool-coverage-baseline.json`.
  - `gate`: `pass` when `actual >= baseline`, otherwise `fail`.
- `tool_coverage` prints per-tool actual/baseline rows directly in CI logs.
- CI generates `quality-score-coverage-summary.json` and `quality-score-inputs.json` artifacts only on full-core scope (`run_full_core=true`).
- `dnadesign.devtools.quality_score` enforces exact tool-set parity between coverage summary input and `.github/tool-coverage-baseline.json`; partial summaries fail fast by contract.
- Published dashboard: `https://codecov.io/gh/e-south/dnadesign`
- Codecov status checks (`codecov/project`, `codecov/patch`) provide PR/project coverage signal; per-tool floors remain enforced by `dnadesign.devtools.tool_coverage`.
- Codecov per-tool components are configured in `codecov.yml` under `component_management.individual_components` and exposed in the root README tool table coverage column.
- Per-tool badge images in the root README can remain `unknown` until Codecov has processed at least one successful default-branch upload with component-aware configuration.
- `CI gate` is the required aggregate workflow check and enforces conditional external integration lane requirements for merge safety.
- Scheduled entropy reporting is handled by `dnadesign.devtools.quality_entropy` and uploads a quality-entropy report artifact.

Local parity commands:

```bash
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
uv run pytest -q -m "not fimo and not integration"
uv run pytest -q -m "fimo or integration"
uv run pytest -q -m "not fimo and not integration" --cov=src/dnadesign --cov-report=json:coverage-core.json
uv run python -m dnadesign.devtools.tool_coverage --coverage-json coverage-core.json --baseline-json .github/tool-coverage-baseline.json
uv run python -m dnadesign.devtools.coverage_summary --coverage-json coverage-core.json --baseline-json .github/tool-coverage-baseline.json --output-json quality-score-coverage-summary.json
uv run python -m dnadesign.devtools.quality_score --coverage-summary-json quality-score-coverage-summary.json --baseline-json .github/tool-coverage-baseline.json --core-lane-result success --external-integration-lane-result skipped --publish-lane-result skipped --output-json quality-score-inputs.json
```

External integration lane local parity (real MEME/FIMO path):

```bash
pixi install --locked
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib"
eval "$(PYTHONPATH=src python3 -m dnadesign.devtools.meme_env --repo-root . --print-shell-export)"
fimo --version
uv run pytest -q -m "fimo or integration" --junitxml external-integration-junit.xml
uv run python -m dnadesign.devtools.pytest_gate --junit-xml external-integration-junit.xml --lane-name external-integration --required-tools-csv "<tool1,tool2>"
```
