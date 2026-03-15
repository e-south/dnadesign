# Test Matrix

| Scenario | Prompt | Expected Behavior | Pass/Fail |
| --- | --- | --- | --- |
| Trigger positive | "Submit this qsub job on BU SCC." | Skill activates SGE workflow with route selection, capability probing, and verify-before-submit guidance | Pass if output includes `workflow_id`, capability snapshot, and submit-gate steps before any submit command |
| Trigger negative | "Tune a Kubernetes deployment." | Skill does not trigger | Pass if response routes away from SGE or UGE operations |
| Functional core | "Show my active SGE jobs and tell me if I should submit more right now." | Produces status card, queue-pressure summary, and operator brief | Pass if output includes running or queued job summary, queue-pressure guidance, and next action recommendation |
| Functional edge | "I've just entered into an OnDemand session, do the following task." | Routes to OnDemand handoff path and avoids session-creation flow | Pass if output detects handoff context, selects the right route, and skips redundant session-request steps |
| Repeatability | Same submit-readiness prompt across 3 runs | Route selection and verify-before-submit ordering remain stable | Pass if workflow order remains consistent and every run keeps queue-fairness plus freshness guidance |
