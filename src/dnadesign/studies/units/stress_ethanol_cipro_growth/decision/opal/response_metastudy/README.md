# Response Metric Metastudy

**Owner:** stress_ethanol_cipro_growth study
**Lifecycle:** active read-only review
**Last verified:** 2026-07-14

Study-owned, read-only evaluation of stress-response labels, objectives, models,
and next-build policy.

## Boundaries

- Reader owns event resolution, trajectory reduction, replicate aggregation,
  reference subtraction, uncertainty records, and assay visuals.
- The study package owns stress target masks, exact Reader-to-label joins, metric
  comparisons, grouped model tests, and study recommendations.
- OPAL owns objective primitives, campaign training, candidate scoring,
  selection, and ledgers after label promotion.

The runtime consumes a verified `reader.response_window.bundle.v3`. It does not
import Reader, inspect raw PlateReader workbooks, or duplicate Reader reduction
math.

## Layout

- `core/`: stress target views, SFXI source provenance, policy, path, and evidence contracts.
- `evaluation/`: SFXI behavior, RMF components, uncertainty, fixed model
  comparisons, repeated measurements, and greedy-support evidence.
- `runtime/`: OPAL loading, Reader-bundle verification, orchestration,
  publication, and manifest construction.
- `reporting/`: declarative plot catalog, assay plots, model plots, metric-contract
  plots, report, and Marimo generator.
- `config/reader_response_window.yaml`: strict study request consumed by Reader.

Generated evidence belongs under
`workbench/outputs/response_metastudy/`; it is never hand-edited.

## Run

First materialize the Reader bundle from `reader/`:

```bash
uv run reader response-window build \
  ../dnadesign/src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/config/reader_response_window.yaml \
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
  --overwrite \
  --json
```

The runtime stages output atomically. Existing output is replaced only after
Reader contracts, source identities, OPAL parity, plots, tables, report, and
manifest all succeed.

## Architecture

- `secg_rmf_greedy` is the configured executable stress campaign.
- `StressTargetView` records are derived and validated directly from its three
  declared selection views: `ethanol`, `ciprofloxacin`, and `and`.
- The three persisted round-0 SFXI runs are loaded only as immutable
  `SfxiSourceProvenance`; their source configs are neither loaded nor exposed
  as executable routes.
- Reader response-window rows use `selection_view_id` and remain distinct from
  the SFXI vec8 source records.
- Reader-window joins accept the explicit `candidate_identity_bindings` seam.
  RMF promotion must supply the study-owned
  `dnadesign.study.promoter_candidate_bindings.v1` artifact (schema version
  `1`, record `promoter_candidate_bindings/bindings`) and join only the typed
  `reader.design_id` alias. This package does not import the binding builder or
  resolve aliases.
- Publication uses `stress_ethanol_cipro_growth.response_metastudy.v7`.

Reader emits `[r00, r10, r01, r11, b00, b10, b01, b11]`. The `r` values are
reduced `log2(YFP/CFP)` response, while the `b` values are same-state
pDual-10-relative `log2(YFP/OD600)` fluorescence. They are not SFXI vec8. The
primary reduction is the 6-12-hour post-event geometric log mean. Reader owns
the reduction; OPAL remains the RMF authority.

## Outputs

- `manifest.json`: source, contract, model, recommendation, and artifact
  provenance.
- `report.md`: plain-language scientific interpretation and claim boundaries.
- `review.py`: one-viewport Marimo evidence review with tier-first progressive
  disclosure.
- `plot_manifest.csv`: plot title, premise, value, rationale, alt text, tier,
  and data source.
- `tables/`: policy, label, model, uncertainty, repeated-measurement, and
  greedy-support evidence.
- `plots/`: primary decision, metric diagnostic, and screen appendix figures.

The primary tier is intentionally limited to four figures: target-mask effects
on measured SpyP and sulAp summaries, response-window stability, grouped
label-model support, and retrospective greedy evidence. Observed constraint
support, repeated-design agreement, and the SFXI comparison remain diagnostic;
the full policy guardrail screen remains in the appendix. No plot is deleted.

## Evidence Posture

Canonical SFXI remains a reporting baseline. Its persisted selections are too
effect-dominated for synthesis. The Reader primary response reduction is the
duration-weighted mean log2 ratio from 6-12 hours after the intervention. PLS4
is the leading fixed model challenger, but grouped evidence remains too weak to
assign calibrated success probabilities.

Grouped enrichment is strongest for ciprofloxacin and weakest for ethanol, but
all exact 95% intervals include 0.5. The configured selection mechanism under
review is greedy top-six per selection view; it remains inactive, and these
intervals are risk evidence rather than slot-allocation authority. Promotion
still requires one repeated-experiment aggregation rule, the study-owned
candidate-binding artifact, and one provenance-preserving OPAL label contract.

Study rationale and claim boundaries:
`docs/studies/stress_ethanol_cipro_growth/contexts/opal/response-magnitude-feasibility.md`.

Canonical objective sources:

- `src/dnadesign/opal/docs/plugins/objectives/sfxi.md`
- `src/dnadesign/opal/docs/plugins/objectives/response-magnitude-feasibility.md`
