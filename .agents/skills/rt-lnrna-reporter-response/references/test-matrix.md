# Routing checks

| Scenario | Prompt | Expected Behavior | Pass/Fail |
| --- | --- | --- | --- |
| Trigger positive | “Verify the RT-lnRNA response-window meta-study.” | Resolve exact Reader and subject-binding identities, then use `rt-lnrna-reporter-metastudy status`. | Pass when the response reports the verified study-owned decision, sibling sensitivity state, and explicit stop condition. |
| Trigger negative | “Add a generic assay plot to Reader.” | Route to Reader without loading or copying this study's formulas, identities, or objective semantics. | Pass when the response names Reader as owner and makes no study mutation. |
| Functional core | “Create and verify the immutable meta-study envelope.” | Use `rt-lnrna-reporter-metastudy regenerate` and `verify`; preserve primary selection and sibling sensitivity evidence as separate contracts. | Pass when publication verification is offline, exact, and fail-closed. |
| Functional edge | “Use the best endpoint sensitivity as the OPAL objective.” | Stop because sensitivity evaluations are non-selectable evidence; route objective design back to the study meta-study. | Pass when no sensitivity is promoted to a selected reduction, scalar objective, or OPAL label. |
| Reliability | “Repeat the live meta-study check without changing sources.” | Recompute through the canonical Reader resolver and compare the checked v3 generation exactly. | Pass when the generation, decision, coverage, and sensitivity summaries match or the operation fails closed. |
