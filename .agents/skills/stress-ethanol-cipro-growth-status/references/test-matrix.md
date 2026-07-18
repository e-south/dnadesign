# Test Matrix

| Scenario | Prompt | Expected Behavior | Pass/Fail |
| --- | --- | --- | --- |
| Trigger positive | Where is stress_ethanol_cipro_growth now? | Use the checked-in status record and OPS JSON provider. | Pass if status JSON parses and the answer reports phase, datasets, and next surface. |
| Trigger negative | Where is regulondb_native_promoter_panel now? | Route away from this study-specific skill. | Pass if the skill does not generalize the stress study record to another study. |
| Functional core | `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json` | Emit record-backed machine-readable status. | Pass if stdout is parseable JSON and `state` is not an unhandled exception. |
| Functional edge | `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --json --command-timeout-seconds 30` | Emit bounded readiness JSON for next-scope blockers. | Pass if stdout is parseable JSON and timeout-bounded checks report explicit statuses. |
| SFXI metric review | Does current SFXI scoring support synthesis? | Route through OPAL to the metastudy context and manifest-backed verdict. | Pass if the answer separates canonical metric correctness, held-out RF support, and biological validation. |
| MSRB campaign status | Is the active MSRB campaign bound to the approved study protocol? | Route through the checked-in record to the MSRB source-of-truth doc, source protocol, activation audit, and `secg_msrb_greedy` config. | Pass if activation, runtime state, model support, and synthesis authorization remain separate claims. |
| Frozen RMF comparator | Is `secg_rmf_greedy` still executable? | Route to the frozen RMF context and workbench source evidence. | Pass if the answer says no and does not offer the comparator config as an OPAL campaign route. |
| Repeatability | Run the skill audit twice. | Structural and routing checks remain deterministic. | Pass if both audit runs finish with no failures and no generated study outputs are required. |
