## References

Progressive-disclosure pack for `sge-hpc-ops`.

Load order:
1. `probe-first-contract.md`
2. `workflow-router.md`
3. `route-load-matrix.md`
4. `session-status-reporting.md` and `user-status-contract.md` when status communication is in scope
5. `submission-shape-advisor.md` and `operator-brief.md` when submit readiness is in scope
6. `interactive-contract.md` or `batch-submit-contract.md` based on selected workflow route
7. `runbook-entrypoints.md` when command-first Ops runbooks are available
8. `automation-qa-preflight.md` for submit workflows
9. `workload-dnadesign.md` when repo context is `dnadesign`
10. `bu-scc-system-usage.md` and `source-evidence.md` when BU SCC policy claims are touched
11. `ci-mechanical-gates.md` when automation/policy gates are requested
