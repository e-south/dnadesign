---
id: stress-ethanol-cipro-growth-response-window-observations
title: Response-window observations
owner: stress_ethanol_cipro_growth
status: review_required
last_verified: 2026-07-15
first_hop: config/observation_policy.yaml
---

# Response-Window Observations

This study package turns verified Reader experiment evidence into candidate-level
response-window observations. It is independent of SFXI, RMF, OPAL model fitting,
and campaign selection.

The ownership chain is:

```text
Reader experiment reductions
  -> exact study candidate bindings
  -> equal-weight experiment evidence
  -> candidate response-window observations
  -> optional OPAL label promotion
```

Reader owns well and time-series reduction. This package owns the scientific
decision about how distinct experiments measuring one candidate are compared
and combined. The candidate-binding artifact owns alias, candidate, and sequence
identity. OPAL consumes a promoted label artifact but does not define repeat
comparability or aggregation.

## Open Next

- `config/reader_response_window.yaml`: the Reader service request, event
  definition, reductions, display vocabulary, and within-experiment quality
  requirements.
- `config/observation_policy.yaml`: exact source-manifest pins, repeat decisions,
  aggregation semantics, uncertainty policy, and study approval.
- `reader_bundle.py`: strict Reader bundle adapter. It imports no Reader code.
- `sources.py`: exact `reader.design_id` resolution through the candidate-binding
  public API.
- `aggregation.py`: candidate point estimates, hierarchical uncertainty,
  reduction sensitivity, and event-time sensitivity.
- `censoring.py`: strict component-bound provenance, bounded-component
  observability, and the exact-label publication gate.
- `repeat_adjudication.py`: typed status, classification, reviewer, timestamp,
  evidence path, and evidence-digest rules for repeated experiments.
- `repeat_diagnostics.py`: component-level experiment minima, medians, maxima,
  and ranges. These describe disagreement; they do not set an acceptance
  threshold.
- `artifact.py`: atomic publication and verification of the complete observation
  bundle.

## Scientific Contract

Each Reader experiment is one evidence unit regardless of its well count. Wells
are not pooled across experiments. Joint Reader bootstrap vectors remain joint
when experiment-level uncertainty is propagated. Reader bootstrap uncertainty,
between-experiment disagreement, alternate-reduction sensitivity, and event-time
sensitivity remain separate evidence lanes.

Bootstrap intervals are explicitly descriptive at the observed experiment
count. With one or two experiments they do not claim calibrated coverage for a
population of future experiments. The bundle records the experiment count,
nominal interval mass, and `population_coverage_claimed=false` on every
component row.

A single experiment passes through unchanged. Exactly two comparable
experiments produce a transparent midpoint, which the artifact names
`two_experiment_midpoint` rather than describing as robust. Three or more
comparable experiments use a component-wise experiment median. The policy does
not support choosing a convenient experiment from a conflicting pair: all
declared experiments must be judged comparable, or the candidate remains
blocked while the source evidence is corrected or remeasured.

Every repeated candidate has one explicit state:

- `review_required`: unresolved and blocks publication;
- `comparable`: included after evidence-backed assay-context review;
- `excluded_noncomparable`: retained in provenance but excluded from the label
  table; or
- `remeasure_required`: retained in provenance and excluded until new evidence
  is available.

A final state requires a controlled classification, named reviewer,
timezone-aware decision time, confined evidence path, and SHA-256 digest. The
policy loader verifies the evidence file and digest. A reviewer cannot convert
`review_required` to `comparable` by changing status text alone.

The value order is:

```text
[r00, r10, r01, r11, b00, b10, b01, b11]
```

The meaning and equations for these fields live in the study's
`contexts/opal/response-magnitude-feasibility.md` and Reader's
`docs/lib/plate_reader/response_window.md`. This package enforces those contracts
without copying objective mathematics.

Publication fails if an exact alias is missing or unexpectedly becomes bound,
a source manifest changes, a repeat gains or loses an experiment, a sequence
identity changes, bootstrap coverage is incomplete, any vector is non-finite,
repeat review remains unresolved, or the study policy lacks named approval.
Reader's per-component bound kind and clipping/overflow causes remain on every
experiment contribution. A non-exact primary component is still visible in the
preview, but it blocks publication as an exact observed label. This package does
not replace, clamp, or impute that value. Supporting bounded labels requires a
separate, explicit censor-aware study policy and contract revision.

The current policy is intentionally `review_required`. A preview is valid
evidence; it is not a candidate-label publication.

## Operator Surface

Run the non-mutating source and policy check from the `dnadesign/` root:

```bash
uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations \
  preview \
  --reader-bundle ../reader/outputs/reviews/stress_response_window/latest \
  --candidate-bindings src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest
```

The JSON result reports the total candidate corpus, the candidate observations
that can be previewed, counts for each repeat-decision state, the maximum
observed component range, bounded primary candidates/contributions/components,
exact source-manifest digests, study approval, and one `ready_to_materialize`
decision. Do not run
`materialize` until the checked-in repeat decisions and named study approval are
complete. `verify` is the read-only entrypoint for an approved bundle.

## Published Bundle

An approved publication uses
`stress_ethanol_cipro_growth.response_window_observations.v1` and contains:

- `manifest.json`
- `observations.parquet`
- `contributions.parquet`
- `hierarchical_bootstrap_draws.parquet`
- `uncertainty.parquet`
- `repeat_diagnostics.parquet`
- `reduction_sensitivity.parquet`
- `event_time_sensitivity.parquet`

The manifest pins the policy, Reader bundle, candidate bindings, value order,
primary reduction, and record digests. Sequence annotations remain in the
candidate-binding artifact; this bundle references candidate identity and does
not create another sequence metadata system. Publication is create-only; a
revised approved corpus uses a new named bundle rather than overwriting evidence
already referenced downstream.
