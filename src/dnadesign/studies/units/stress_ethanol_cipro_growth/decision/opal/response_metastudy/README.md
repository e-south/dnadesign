---
id: stress-ethanol-cipro-growth-response-metastudy-package
title: Response metric metastudy package
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-14
---

# Response Metric Metastudy

Study-owned, read-only evaluation of response-window Y, objectives, models, and
next-build policy.

## Boundaries

- Reader owns event resolution, trajectory reduction, replicate aggregation,
  reference subtraction, uncertainty records, and assay visuals.
- The study package owns stress target masks, exact Reader-to-label joins,
  objective-specific evaluation, grouped model tests, and study recommendations.
- OPAL owns objective primitives, campaign training, candidate scoring,
  selection, and ledgers after observed-Y promotion.

The runtime consumes a verified `reader.response_window.bundle.v4` produced
from `reader.response_window.request.v3`. Both carry the stress `study_id`. The
runtime does not import Reader, inspect raw PlateReader workbooks, or duplicate
Reader reduction math.

## Layout

- `core/`: stress target views, SFXI source provenance, policy, path, and evidence contracts.
- `evaluation/`: independent SFXI source-evidence review, RMF components,
  uncertainty, fixed model comparisons, repeated measurements, and
  greedy-support evidence.
- `runtime/`: OPAL loading, Reader-bundle verification, orchestration,
  publication, and manifest construction.
- `reporting/`: declarative plot catalog, assay plots, model plots, metric-contract
  plots, report, and Marimo generator.
- `config/reader_response_window.yaml`: strict study request consumed by Reader.
- `config/response_model_screen_selection.yaml`: exact Reader experiment/design
  pairs used only by the retrospective response model screen. It contains no
  candidate IDs, accounts explicitly for Reader designs that have no study
  candidate binding, and defines no promotion aggregation.

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
  --candidate-bindings src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest \
  --overwrite \
  --json
```

The runtime stages output atomically. Existing output is replaced only after
Reader contracts, source identities, OPAL parity, plots, tables, report, and
manifest all succeed. It also rejects campaign RMF thresholds or scales that
drift from the Reader-derived review calibration beyond six-decimal rounding.

## Architecture

- `secg_rmf_greedy` is the configured executable stress campaign.
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
  choice for repeated designs is screen-only and is not a repeat-aggregation
  rule for observed-label promotion.
- Publication uses `stress_ethanol_cipro_growth.response_metastudy.v9`.

Reader emits `[r00, r10, r01, r11, b00, b10, b01, b11]`. The `r` values are
reduced `log2(YFP/CFP)` response, while the `b` values are same-state
pDual-10-relative `log2(YFP/OD600)` fluorescence. They are not SFXI vec8. The
eight-component vector is response-window Y, not an RMF vec8. The primary
reduction is the 6-12-hour post-event geometric log mean. Reader owns the
reduction; OPAL applies the RMF objective. Promotion of response-window
observed labels remains inactive.

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
support and repeated-design agreement remain diagnostic; the full SFXI source-
evidence policy screen remains in the appendix.

## Evidence Posture

SFXI source evidence is evaluated only under its declared vec8 contract.
Response-window evidence is evaluated only from its declared `r` and `b`
fields; the study does not translate those fields into an SFXI vector. The SFXI
source selections are too effect-dominated for synthesis. The Reader primary
response reduction is the duration-weighted mean log2 ratio from 6-12 hours
after the intervention. PLS4 is the leading fixed model challenger, but grouped
evidence remains too weak to assign calibrated success probabilities.

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
