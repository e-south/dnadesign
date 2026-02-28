## Quality Docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-18

### At a glance
This index tracks quality expectations for tests, CI signal, coverage gates, and documentation parity.

### Contents
- [Root quality score (SOR)](../../QUALITY_SCORE.md): grading model, evidence rules, scorecard, and gap tracker.
- [CI workflow](../../.github/workflows/ci.yaml): lane semantics and gating contracts.
- [Devtools modules](../../src/dnadesign/devtools): executable checks for docs, scope, and coverage.
- [Coverage dashboard](https://codecov.io/gh/e-south/dnadesign): project and PR diff coverage signal.
- [Codecov configuration](../../codecov.yml): status-check policy for project and patch coverage.
- [Codecov per-tool components](../../codecov.yml): `component_management.individual_components` maps coverage by tool path.
- [Codecov component status defaults](../../codecov.yml): `component_management.default_rules.statuses` enforces strict project-status semantics for every component.
- [CI upload wiring](../../.github/workflows/ci.yaml): core/external integration lane uploads use `codecov/codecov-action@v5` with GitHub OIDC authentication (`use_oidc: true`).
- [Coverage baselines](../../.github/tool-coverage-baseline.json): required per-tool floors.
- [Coverage summary generator](../../src/dnadesign/devtools/coverage_summary.py): builds `quality-score-coverage-summary.json` from coverage + baseline contracts.
- [Quality score input generator](../../src/dnadesign/devtools/quality_score.py): composes lane outcomes + coverage summary into `quality-score-inputs.json`.
- [Quality entropy report artifact](../../.github/workflows/ci.yaml): scheduled report for stale SOR metadata and broken evidence links.
- [Developer docs](../dev/README.md): maintainer-level test and CI conventions.
