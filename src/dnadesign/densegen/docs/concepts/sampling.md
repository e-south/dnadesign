## Sampling model

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This concept page explains how DenseGen selects motif candidates before solving, across both Stage-A pool construction and Stage-B library sampling. Read it when you need to tune diversity, reduce failures, or reason about sampling-related runtime behavior.

### Two-stage sampling model
This section defines the boundary between Stage-A and Stage-B responsibilities.

- Stage-A transforms configured inputs into retained pools (`outputs/pools/`).
- Stage-B builds per-plan libraries from those pools (`outputs/libraries/`).

A useful operator rule is to tune Stage-A pool quality first, then tune Stage-B library composition.

For hands-on commands that exercise both stages, use the **[sampling baseline tutorial](../tutorials/demo_sampling_baseline.md)**.

### Stage-A selection rules
This section summarizes the high-signal Stage-A knobs and what they control.

- `sampling.mining.budget.candidates` controls how many candidates are evaluated.
- `sampling.n_sites` controls how many candidates are retained.
- `sampling.selection.policy` controls ranking/diversity strategy.
- `sampling.uniqueness.*` controls dedupe behavior and collision strictness.

For PWM-backed inputs, Stage-A includes mining, scoring, dedupe, and selection before writing pool artifacts.

### Stage-B library rules
This section describes how Stage-B builds plan-scoped libraries from Stage-A outputs.

- `generation.sampling.pool_strategy` controls whether Stage-B uses full pools or subsampled libraries.
- `generation.sampling.library_size` controls per-plan library breadth.
- `generation.plan[*].sampling.include_inputs` controls which pools each plan can draw from.

### Stage-B weighting
This section documents the weighting contract used for coverage balancing and failure avoidance.

When coverage boosting is enabled, under-used candidates receive higher selection weight:

- `weight = 1 + alpha / (1 + count)^power`

When failure avoidance is enabled, repeatedly failing candidates are down-weighted:

- `weight /= (1 + failure_alpha * fails)^failure_power`

These weights apply only to Stage-B library sampling and do not change Stage-A pool contents.

### Strand and scoring notes
This section clarifies why strand interpretation can differ between scoring and final validation.

Stage-A scoring behavior depends on the configured scoring backend, while final acceptance is controlled by sequence constraints at final-sequence validation time. Use **[pipeline lifecycle](pipeline-lifecycle.md)** for the lifecycle view and **[observability and events](observability_and_events.md)** for diagnostics/event boundary context.

### Where to tune first
This section gives a pragmatic tuning order for common quality issues.

1. Increase Stage-A mining budget before increasing retained pool size.
2. Tighten Stage-A quality filters before widening Stage-B library size.
3. Tune Stage-B weighting only after pool quality and plan feasibility are acceptable.

For upstream motif preparation details, use **[Cruncher to DenseGen PWM handoff](../howto/cruncher_pwm_pipeline.md)**.
