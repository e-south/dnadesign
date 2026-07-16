---
id: stress-ethanol-cipro-growth-response-window-observations
title: Response-window observations
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-15
first_hop: config/observation_policy.yaml
---

# Response-window observations

This package turns verified Reader experiment reductions into candidate-level
response-window observations for the stress promoter study. It does not define
SFXI, RMF, OPAL models, or campaign selection.

The handoff is:

```text
Reader experiment reductions
  -> exact study candidate bindings
  -> explicit study label-source decisions
  -> candidate response-window observations
  -> optional OPAL label promotion
```

Reader owns wells, trajectories, event alignment, and within-experiment
reductions. The stress study owns candidate identity, repeat disposition, and
the exact experiment used as each candidate label. OPAL consumes a verified
label publication. It does not choose among Reader experiments.

## Open next

- `config/reader_response_window.yaml`: Reader request, 4-8-hour primary
  reduction, sensitivity reductions, display vocabulary, and assay checks.
- `config/observation_policy.yaml`: source-manifest pins, exact label-source
  decisions, censoring policy, and study approval.
- `config/evidence/repeat_adjudication_4_8h_v1.json`: typed comparison evidence
  for every repeated candidate.
- `reader_bundle.py`: strict Reader bundle adapter with no Reader imports.
- `sources.py`: exact Reader alias resolution through the study candidate
  binding public API.
- `label_sources.py`: one explicit label source per eligible candidate while
  retaining every experiment as evidence.
- `uncertainty.py`: joint Reader bootstrap propagation from the selected
  source.
- `repeat_diagnostics.py`: component ranges across every measured source.
- `censoring.py`: bound provenance and the exact-label gate.
- `artifact.py`: create-only publication and full bundle verification.

## Scientific contract

The response vector is:

```text
[r00, r10, r01, r11, b00, b10, b01, b11]
```

The primary reduction is the duration-weighted geometric log mean from 4 to 8
hours after the intervention. Reader defines the reduction and field meanings.
The study applies candidate identity and source decisions without recomputing
Reader values.

One Reader experiment is one evidence unit regardless of well count. Wells are
never pooled across experiments. A candidate measured once passes through
unchanged. A repeated candidate must have one explicit disposition:

- `label_source_selected`: use the declared Reader experiment as the label source;
- `label_source_excluded`: retain all evidence but publish no label;
- `remeasure_required`: retain all evidence and wait for a new experiment; or
- `review_required`: block publication until the study decides.

All measured experiments remain in `contributions.parquet`. A
`label_source_selected`
repeat has exactly one label-source contribution. This makes “use the most
recent reviewed experiment” an auditable study decision rather than an
inferred ordering rule. Reader experiment slugs are not treated as timestamps
by code.

The selected source supplies the point estimate, joint bootstrap draws,
alternate-reduction sensitivity, and event-time sensitivity. Cross-experiment
ranges remain a separate evidence lane. Selected-source bootstrap intervals are
descriptive and set `population_coverage_claimed=false`; they do not model a
population of future experiments or promoter-cell heterogeneity.

Every primary component must be exact. A selected source with a clipped or
overflow-bounded component remains visible but receives no candidate label.
The package never imputes a bounded value and never falls back to a different
experiment automatically.

The typed repeat-evidence file binds the Reader manifest digest, primary
reduction, candidate ID, exact experiment set, selected source or exclusion,
status, classification, and all eight component ranges. It validates evidence
identity but does not encode a universal disagreement cutoff.

The current approved policy selects the newest reviewed source for eight
repeated candidates. ES22, ES25, ES28, and ES30 are excluded because unresolved
source disagreement prevents one defensible label. ES26 selects its newest
source, but exact-only censoring excludes that bounded source from labels.
“Unresolved source disagreement” does not claim biological heterogeneity,
technical failure, or assay-context drift; the present bulk data do not
distinguish those causes.

## Failure conditions

Publication fails on any unresolved repeat, missing or ambiguous alias, source
digest drift, candidate or sequence mismatch, undeclared experiment, malformed
repeat evidence, incomplete bootstrap coverage, non-finite vector, or missing
study approval. Artifact verification recomputes selected-source values,
one-source contribution flags, uncertainty, sensitivities, and record digests.

Candidate bindings remain the study source of truth for Reader aliases,
candidate IDs, sequence identity, and BaseRender metadata. This package carries
the resolved identity needed for evidence; it does not create another sequence
metadata system.

## Operator surface

Preview without writing:

```bash
uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations \
  preview \
  --reader-bundle ../reader/outputs/reviews/stress_response_window/latest \
  --candidate-bindings src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest
```

Publish an approved immutable bundle:

```bash
uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations \
  materialize \
  --reader-bundle ../reader/outputs/reviews/stress_response_window/latest \
  --candidate-bindings src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest \
  --out-dir src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/response_window_observations/4_8h_v1 \
  --allowed-output-root src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/response_window_observations
```

An approved `stress_ethanol_cipro_growth.response_window_observations.v2`
bundle contains:

- `manifest.json`
- `observations.parquet`
- `contributions.parquet`
- `bootstrap_draws.parquet`
- `uncertainty.parquet`
- `repeat_diagnostics.parquet`
- `reduction_sensitivity.parquet`
- `event_time_sensitivity.parquet`

The manifest pins the policy, Reader bundle, candidate bindings, value order,
primary reduction, and every record digest. Publication is create-only; a new
scientific decision receives a new named bundle.
