# Performance Budgets

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-17

The package currently treats promoter-study scale as a normal operating target, but checked-in verification remains contract-first and fixture-scale.

Current smoke coverage:

- `tests/contracts/test_study_workspace_contracts.py`
- `tests/contracts/test_promoter_reference_margin_template.py`
- `tests/cli/test_workspace_command.py`

Registered smoke slices:

- workspace snapshot publication
- canonical browser geometry inventory
- study-bound workspace initialization from promoter-study bindings
- deep workspace validation for the checked-in promoter template

Each smoke record emits:

- correctness summary
- contract-schema validation
- artifact-presence and docs-reference checks

Interpretation:

- these fixture-scale checks are regression guards for the refactored study seam and artifact contracts
- they are not substitutes for live promoter-study pressure runs on the real USR planes
- live study workspaces that rely on large USR directory overlays should opt into `digest_ledger.json` source provenance before regeneration when snapshot freshness latency becomes dominant
- the April 17, 2026 live `stress_ethanol_cipro_growth` snapshot pressure run measured warmed `workspace_snapshot()` at 1.12s mean across three runs after shared overlay-inventory freshness caching, down from 2.36s cumulative on the profiled pre-pass hot path
- the same study now measures warmed `workspace_snapshot()` at 0.22s mean across three runs after decoupling browser geometry assembly from full notebook context-audit work, with a fresh-process single run at 1.05s
- after splitting plot-semantics sidecar validation out of generic workspace loading and caching the canonical `output_root` on `WorkspaceContext`, the same study now measures warmed `workspace_snapshot()` at 0.18s mean across three runs and warmed `deliverable_status(representation_health_summary)` at 0.13s mean across three runs
- fresh-process `workspace_snapshot()` launches after that loader/control-plane split measured 1.75s on the first cold launch and about 1.00s on subsequent independent launches
- the profiled status path is now dominated by workspace-config YAML parsing plus deliverable freshness recursion rather than plot-semantics sidecar validation
- real-study pressure evidence should continue to be recorded in the development journal

See also:

- [../dev/journal.md](../dev/journal.md)
- [../workflows/promoter-study-representation-comparison.md](../workflows/promoter-study-representation-comparison.md)
