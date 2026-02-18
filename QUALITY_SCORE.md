# QUALITY SCORE

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-18

## At a glance
This document defines how `dnadesign` grades quality across tool domains and architectural layers.
Scores are only valid when backed by evidence links and an explicit next action.

## Contents
- [Scoring model](#scoring-model)
- [Evidence standard](#evidence-standard)
- [Quality scorecard](#quality-scorecard)
- [Generated score inputs](#generated-score-inputs)
- [Mechanical enforcement coverage](#mechanical-enforcement-coverage)
- [Gap tracker](#gap-tracker)
- [Entropy control cadence](#entropy-control-cadence)
- [Autonomy readiness](#autonomy-readiness)
- [References](#references)

## Scoring model
Quality is graded in two axes:
- Domains: tool packages, shared CI/devtools, and operations/docs.
- Layers: correctness, contract enforcement, operability/observability, and documentation fidelity.

Each score uses a `0-4` rubric:
- `0`: missing contract or broken behavior.
- `1`: ad-hoc/manual behavior with high drift risk.
- `2`: baseline behavior exists but with known gaps.
- `3`: contract is stable and enforced in CI.
- `4`: contract is stable, enforced, and trend is improving.

## Evidence standard
- Every scored row must include at least one evidence link (CI workflow, test suite, runbook, or dashboard).
- Every scored row must include `Owner`, `Last verified`, and `Next action`.
- Rows without evidence or verification metadata are treated as stale regardless of numeric value.

## Quality scorecard
| Area | Axis | Score (0-4) | Trend | Gate | Evidence | Owner | Last verified | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ci-devtools` | contract enforcement | 3 | stable | enforced | `.github/workflows/ci.yaml`; `src/dnadesign/devtools` | dnadesign-maintainers | 2026-02-18 | keep scope and boundary checks test-backed |
| `tool-coverage` | correctness floor | 3 | stable | enforced | `.github/tool-coverage-baseline.json`; `https://codecov.io/gh/e-south/dnadesign` | dnadesign-maintainers | 2026-02-18 | raise non-zero baselines for currently unmeasured tools |
| `docs-sor` | documentation fidelity | 3 | improving | enforced | `src/dnadesign/devtools/docs_checks.py`; `docs/exec-plans/README.md` | dnadesign-maintainers | 2026-02-18 | expand owner granularity to per-domain maintainers |

## Generated score inputs
- Canonical score inputs are generated in CI only for full-core scope (`run_full_core=true`) from:
  - core-lane coverage artifact: `coverage-core.json` (per-tool baseline gate input)
  - core-lane summary artifact: `quality-score-coverage-summary.json`
  - generated quality input artifact: `quality-score-inputs.json`
  - core/external integration Codecov uploads: `coverage-core.xml`, `coverage-external-integration.xml` (when coverage scope is active)
  - baseline contract: `.github/tool-coverage-baseline.json`
  - workflow lane outcomes: detect/core/external-integration/quality-score-inputs/ci-gate job results
- On scoped PR runs (`run_full_core=false`), CI enforces per-tool coverage gates and uploads core-lane coverage, but skips canonical quality-score artifact generation by contract.
- Published signal endpoints: `https://codecov.io/gh/e-south/dnadesign`, `codecov/project`, `codecov/patch`
- Manual narrative in this doc explains interpretation and improvement priorities; baseline enforcement remains CI-executable in-repo.

## Mechanical enforcement coverage
| Contract | Enforcement path | Status |
| --- | --- | --- |
| Docs naming/link integrity | `dnadesign.devtools.docs_checks` + CI | enforced |
| Core vs external integration test semantics | pytest markers + CI lanes | enforced |
| External integration non-skipped execution (per in-scope external integration tool) | `dnadesign.devtools.pytest_gate` + JUnit XML in CI | enforced |
| Per-tool coverage floors | `dnadesign.devtools.tool_coverage` + baseline JSON | enforced |
| Quality score input generation | `dnadesign.devtools.coverage_summary` + `dnadesign.devtools.quality_score` in CI | enforced |
| Tool-inventory alignment | `dnadesign.devtools.ci_changes` contracts | enforced |
| Root README tool catalog integrity | `dnadesign.devtools.docs_checks` tool table + path checks | enforced |
| Selected runbook metadata | `dnadesign.devtools.docs_checks` + CI | enforced |

## Gap tracker
| Gap | Impact | Tracking artifact | Exit criteria |
| --- | --- | --- | --- |
| Tools with `0.0` baseline coverage | low confidence on behavior changes | `.github/tool-coverage-baseline.json` | each tool has non-zero baseline from real tests |
| Runbook metadata outside selected enforce-set remains manual | stale runbook ownership/freshness risk | `docs/**/*.md` outside enforced runbook set | extend docs checks to additional runbook classes as they become operationally critical |
| Partial score ownership granularity | unclear accountability | this file | each score row has named owner role/team |

## Entropy control cadence
- Per PR: CI enforces docs checks, core tests, coverage gate, and external integration lane when in scope.
- Weekly: review scorecard trend and close one tracked gap.
- Monthly: prune stale docs links and refresh `Last verified` timestamps.
- Release cut: confirm scorecard evidence links still resolve and match shipped behavior.

## Autonomy readiness
Autonomy readiness is the repo's ability to let agents execute safely with predictable outcomes.
Current posture:
- Strong: CI/devtools contracts are executable and fail fast.
- Medium: operational runbooks are rich but metadata ownership is still maturing.
- Priority: reduce manual-only quality controls by turning repeatable checks into CI-enforced contracts.

## References
- Harness engineering principles (system-of-record and quality grading): `https://openai.com/index/harness-engineering/`
- CI workflow: `.github/workflows/ci.yaml`
- Devtools checks/gates: `src/dnadesign/devtools/`
- Maintainer CI/test guide: `docs/dev/README.md`
- Coverage dashboard: `https://codecov.io/gh/e-south/dnadesign`
