---
id: stress-ethanol-cipro-growth-response-metastudy-package
title: Response assay and objective comparison package
owner: stress_ethanol_cipro_growth
status: source_evidence
last_verified: 2026-07-20
---

# Response assay and objective comparison

Study-owned, read-only evaluation of response-window Y, objectives, models, and
next-build policy.

## Boundaries

- Reader owns event resolution, trajectory reduction, within-experiment well
  summaries, reference subtraction, uncertainty records, and assay visuals.
- The study package owns stress target masks, exact Reader-to-label joins,
  repeated-candidate label-source decisions, objective-specific evaluation,
  grouped model tests, and study recommendations.
- OPAL owns objective primitives, campaign training, candidate scoring,
  selection, and ledgers after observed-Y promotion.

The runtime consumes a verified `reader.response_window.bundle.v5` produced
from `reader.response_window.request.v3`. Both carry the stress `study_id`. The
runtime does not import Reader, inspect raw PlateReader workbooks, or duplicate
Reader reduction math.

The window comparison may read Reader-published well and trajectory records for
pDual-10 replicate spread, growth/OD context, and measurement observability.
Those records are diagnostic only: the study never recomputes or substitutes
Reader-owned response-window Y.

## Layout

- `core/`: stress target views, SFXI source provenance, policy, path, and evidence contracts.
- `evaluation/`: independent SFXI source-evidence review, RMF components,
  uncertainty, fixed model comparisons, repeated measurements, and
  greedy-support evidence.
- `runtime/`: OPAL loading, Reader-bundle verification, orchestration,
  publication, and manifest construction.
- `model_evidence/`: frozen evaluation protocols plus immutable scientific
  checkpoints for the model-evidence trajectory. OPAL run progress is outside
  this package.
- `reporting/`: declarative plot catalog, assay plots, model plots, metric-contract
  plots, report, and Marimo generator.
- `config/response_model_screen_selection.yaml`: exact Reader experiment/design
  pairs used only by the retrospective response model screen. It contains no
  candidate IDs, accounts explicitly for Reader designs that have no study
  candidate binding, and has no label-truth role.
- `config/multistate_response_behavior_shadow_v1.yaml`: persisted, shadow-only
  binding for target masks, the shared soft-min scale recipe, evidence roles,
  and activation gates. It does not configure a campaign.

The shadow behavior modules expose a bounded builder for observed rows, Reader
joint-bootstrap draws, and fixed prediction matrices. They use OPAL's public
scorer and emit digest-bearing scores, coordinates, event envelopes, repeat
agreement, observed candidate-experiment-unit rank sensitivity, and
hard-versus-smooth candidate rank evidence on the fixed prediction surface.
They do not alter the normal metastudy publication or campaign state. Their
allocation comparison is a read-only call to OPAL's public
sequence-deduplicated runtime.

Use `multistate_behavior_cli.py preview` for a read-only summary, `publish` for
the atomic shadow bundle, and `verify` for a fail-closed artifact check. The
publisher preserves scale-derivation rows and bootstrap scores rather than
retaining only derived summaries. It also emits normalization sensitivity,
grouped behavior-versus-RMF prediction-to-truth validation, corrected-Reader
RMF replay scales, fixed raw prediction vectors, sequence-unique allocation
previews, a digest-bound split decision, an independent adversarial audit,
`report.md`, and three minimal review plots.

To reproduce the current shadow bundle without changing its prediction source,
read the pinned run ID from the existing manifest and pass it back to the
publisher:

```bash
BUNDLE=src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/multistate_response_behavior_shadow/latest
PREDICTION_RUN_ID=$(jq -r '.source.prediction.run_id' "$BUNDLE/manifest.json")
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.multistate_behavior_cli publish \
  --reader-bundle ../reader/outputs/reviews/stress_response_window/latest \
  --candidate-bindings src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest \
  --prediction-run-id "$PREDICTION_RUN_ID" \
  --out-dir "$BUNDLE" \
  --overwrite
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.multistate_behavior_cli verify \
  --bundle "$BUNDLE"
```

Publication stages the complete bundle and replaces the destination only after
its tables, figures, report, manifest, and digests verify.

The objective-neutral response-window Reader request and candidate-observation
policy live in the study-level `response_window_observations/` package. The metastudy consumes
their verified output; it does not own assay reduction or label truth.

The main response metastudy bundle belongs under
`workbench/outputs/response_metastudy/`. The separate multistate-behavior
shadow bundle belongs under
`workbench/outputs/multistate_response_behavior_shadow/`. Neither publication
is hand-edited.

## Run

First materialize the Reader bundle from `reader/`:

```bash
uv run reader response-window build \
  ../dnadesign/src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/config/reader_response_window.yaml \
  --reader-root . \
  --out-dir outputs/reviews/stress_response_window/latest \
  --overwrite \
  --format json
```

Then run the metastudy from `dnadesign/`:

```bash
uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy \
  --reader-bundle ../reader/outputs/reviews/stress_response_window/latest \
  --candidate-bindings src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest \
  --calibration-preview \
  --json
```

The preview is read-only. It derives assay scales from every exact, study-bound,
non-reference primary Reader candidate-experiment row. This declared cohort is
independent of the retrospective model-screen selection and repeated-candidate
label decisions. The output records the cohort rule, Reader and candidate-binding
digests, counts, target masks, exact derived values, and six-decimal campaign
parity.

Publish the complete review bundle:

```bash
uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy \
  --reader-bundle ../reader/outputs/reviews/stress_response_window/latest \
  --candidate-bindings src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest \
  --overwrite \
  --json
```

The runtime stages output atomically. Existing output is replaced only after
Reader contracts, source identities, plots, tables, report, and manifest all
succeed. Its model-screen calibration remains diagnostic. A difference between
that screen and the declared campaign cohort is reported and does not redefine
the campaign scale contract.

## Architecture

- `secg_rmf_greedy` is frozen non-executable comparator evidence under the
  study workbench source-evidence shelf.
- `StressTargetView` records are derived and validated directly from its three
  declared selection views: `ethanol`, `ciprofloxacin`, and `and`.
- The three persisted round-0 SFXI runs are loaded only as immutable
  `SfxiSourceProvenance`; their source configs are neither loaded nor exposed
  as executable routes.
- Reader response-window rows use `selection_view_id` and remain distinct from
  the SFXI vec8 source records.
- Reader-window joins require the study-owned
  `dnadesign.study.promoter_candidate_bindings.v1` artifact (schema version
  `1`, record `promoter_candidate_bindings/bindings`). The response-owned
  screen selection declares exact Reader experiment/design pairs; the runtime
  resolves their candidate IDs only through exact `reader.design_id` aliases.
  It consumes the public verifier and loader, not the binding builder. Every
  non-reference design in the Reader primary reduction must be selected once
  or declared as absent from the study binding artifact; the runtime rejects
  silent omissions and exclusions that later become resolvable.
- The response model screen does not read SFXI source CSVs. Its explicit source
  choice for repeated designs is screen-only and has no label-source or
  calibration-cohort authority.
- Publication uses `stress_ethanol_cipro_growth.response_metastudy.v12`.

Reader emits `[r00, r10, r01, r11, b00, b10, b01, b11]`. The `r` values are
reduced `log2(YFP/CFP)` response, while the `b` values are same-state
pDual-10-relative `log2(YFP/OD600)` fluorescence. They are not SFXI vec8. The
eight-component vector is response-window Y, not an RMF vec8. The primary
reduction is the 4-8-hour post-event geometric log mean. Reader owns the
reduction. This package applies RMF only as a historical comparator; the active
OPAL campaign applies MSRB. Response-window labels enter OPAL only through the
study's verified, manifest-pinned publication contract.

Every active view uses global target-state separation: the least responsive ON
state is compared with the most responsive OFF state under the declared mask.
Conditional induction and factorial interaction remain separately named
diagnostics and are not alternate names for RMF.

## Outputs

- `manifest.json`: source, contract, model, recommendation, and artifact
  provenance.
- `report.md`: plain-language scientific interpretation and claim boundaries.
- `review.py`: one-viewport Marimo review organized by assay and labels,
  historical model screens, RMF comparator evidence, and SFXI comparator
  evidence.
- `plot_manifest.csv`: plot title, premise, value, rationale, alt text, review
  section, storage tier, and data source.
- `tables/`: policy, label, model, uncertainty, repeated-measurement, and
  greedy-support evidence.
- `plots/`: primary decision, metric diagnostic, and screen appendix figures.
- `model_evidence/`: optional immutable checkpoints projected from verified
  metastudy bundles; see `model_evidence/README.md`.

The default review section begins with response-window stability, followed by
event timing and repeated-design agreement. Historical SFXI and response-window
model screens remain separate from RMF and SFXI comparator sections; they are
not direct validation of the active MSRB campaign. Storage tiers continue to
place files under primary, diagnostic, and appendix directories, but they are
not the user-facing navigation ontology.

## Evidence Posture

SFXI source evidence is evaluated only under its declared vec8 contract.
Response-window evidence is evaluated only from its declared `r` and `b`
fields; the study does not translate those fields into an SFXI vector. The SFXI
source selections are too effect-dominated for synthesis. The Reader primary
response reduction is the duration-weighted mean log2 ratio from 4-8 hours
after the intervention. The configured campaign random forest is evaluated as
the campaign model. Fixed challengers and the mean baseline remain separate
comparators; none is promoted by the present grouped evidence.

Grouped enrichment is strongest for ciprofloxacin and weakest for ethanol, but
all exact 95% intervals include 0.5. Round 0 used the configured RF and a
deterministic round-robin allocator to assign six sequence-unique slots per
view. It is not scientifically promoted, and the intervals remain risk evidence
rather than slot-allocation authority. The candidate-binding artifact and
explicit label-source policy are declared and approved. The generated manifest
reports typed-label readiness separately from model support, selection-policy
promotion, and synthesis authorization.

The model-evidence trajectory records the same frozen grouped screen after each
eligible corpus update. A protocol change begins a separate series. Current
checkpoints are retrospective and nonpromoted; prospective evidence begins only
when predictions are fixed before the corresponding measurements are observed.

After each measured batch, record one checkpoint only after Reader evidence,
candidate bindings, repeat adjudication, and the metastudy bundle verify. Review
candidate count, channel-level X-to-Y rank preservation and error, the weakest
selection-view ordering, group support, and configured-campaign greedy support
together. Improvement is not required to be monotonic at low sample counts;
the immutable series makes regressions and uncertainty visible instead of
silently replacing an earlier screen. Challenger-model progress remains
descriptive and cannot satisfy the configured campaign-model gate.

Study rationale and claim boundaries:
`docs/studies/stress_ethanol_cipro_growth/contexts/opal/response-magnitude-feasibility.md`.

Canonical objective sources:

- `src/dnadesign/opal/docs/plugins/objectives/sfxi.md`
- `src/dnadesign/opal/docs/plugins/objectives/response-magnitude-feasibility.md`
- `src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md`

Stress-study binding and claim boundary:

- `docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior.md`
