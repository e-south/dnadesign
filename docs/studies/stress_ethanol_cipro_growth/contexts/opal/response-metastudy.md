---
id: stress-ethanol-cipro-growth-opal-response-metastudy
title: Response metric metastudy
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-14
audience:
  - scientist
  - maintainer
  - agent
---

## Response Metric Metastudy

**Status:** round-0 metric, label, and predictor review
**Owner:** `stress_ethanol_cipro_growth` study
**Last verified:** 2026-07-14
**Implementation:** `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy`
**Generated evidence:** `src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/response_metastudy/latest`

### Premise

The study should promote response-window observed Y and activate an
objective-selector pairing only when the assay summary is reproducible,
changing the stress target view changes ranking as intended, and X preserves
useful ordering in held-out Reader experiments.

### Scope

The metastudy compares two distinct questions:

1. Why did canonical SFXI produce correlated target-view selections?
2. Can event-relative response and reference-relative fluorescence support a
   more direct target-margin objective and a defensible next-build strategy?

It is read-only with respect to Reader records, OPAL labels, campaign configs,
ledgers, and synthesis handoffs.

### Ownership

- Reader owns event resolution, trajectory reduction, replicate aggregation,
  joint bootstrap draws, reference subtraction, and assay review visuals.
- The stress study owns target masks, Reader-to-candidate joins, label
  representation comparisons, grouped model evaluation, and promotion policy.
- OPAL owns canonical SFXI, Response-Magnitude Feasibility (RMF) math, model fitting,
  candidate scoring, selection, and ledgers after promotion.

The metastudy consumes Reader's public bundle. It does not import Reader or
maintain duplicate trajectory math.

### Campaign, target-view, and source boundaries

`secg_rmf_greedy` is the configured executable stress campaign. The
metastudy loads its campaign config and derives exactly three immutable
`StressTargetView` records in declared order: `ethanol`, `ciprofloxacin`, and
`and`. Each view must use `response_magnitude_feasibility_v1`, state order
`[00, 10, 01, 11]`, and a unique binary target mask. The loader rejects extra,
missing, reordered, or malformed views.

The three persisted SFXI runs are not executable campaign definitions. They
enter as `SfxiSourceProvenance` values containing a source ID, the recorded
source campaign slug, exact run ID, and target-view ID. The runtime reads their
round-0 ledgers and measured Reader source rows directly; it does not load the
SFXI source campaign configs or expose them as executable routes.
Output tables identify the configured masks with `selection_view_id`.

Reader-window joins require the study-issued artifact with schema ID
`dnadesign.study.promoter_candidate_bindings.v1`, schema version `1`, study ID
`stress_ethanol_cipro_growth`, and record
`promoter_candidate_bindings/bindings`. Its typed alias key is
`(alias_namespace, alias)`. The response-owned
`response_model_screen_selection.yaml` declares exact Reader experiment/design
pairs without candidate IDs. The metastudy resolves each candidate only through
the `reader.design_id` binding, rejects missing or duplicate resolution, and
does not import the binding builder.

The selection is limited to retrospective model screening. It makes the source
choice for repeated designs explicit, does not read an SFXI source CSV, and does
not define the repeat aggregation required for observed-label promotion.

The response-metastudy publication schema is
`stress_ethanol_cipro_growth.response_metastudy.v9`.

### Evidence Flow

1. Verify the three immutable round-0 SFXI source ledgers, shared 35-row label pool,
   candidate IDs, and their equivalent predictor surfaces.
2. Recompute persisted SFXI scores through the public OPAL API.
3. Audit SFXI exponent, gate, lexicographic, and OFF-state-logic variants without
   changing `secg_rmf_greedy` campaign state.
4. Verify `reader.response_window.bundle.v4`, all record contracts, source
   provenance, artifact digests, and row counts.
5. Verify the response-owned 35-row screen selection against the Reader bundle,
   resolve each design alias through the study binding artifact, and join by
   exact Reader experiment and design identity.
6. Evaluate response and fluorescence requirement stability across five
   Reader-owned event-relative reductions.
7. Apply ethanol, ciprofloxacin, AND, and OR pressure-test masks to raw Reader
   state summaries and joint bootstrap draws.
8. Compare fixed mean, robust RF, fold-fitted PCA-ridge, and PLS challengers with
   complete Reader experiments held out.
9. Measure repeated-design agreement, retrospective enrichment, and the risk of
   a prespecified greedy top-six policy.
10. Publish typed tables, a manifest-backed plot catalog, a report, and one
    Marimo review notebook.

The minimal review path is Reader summary -> target mask -> three raw
requirements -> one maximin feasibility margin -> grouped model check. SFXI
parameter sweeps, overlap screens, and alternative reductions remain diagnostic
because none can replace that path.

### Canonical Sources

Canonical SFXI mathematics:

- `src/dnadesign/opal/docs/plugins/objectives/sfxi.md`
- `src/dnadesign/opal/src/objectives/sfxi_math.py`
- `src/dnadesign/opal/src/objectives/sfxi_v1.py`

Response-Magnitude Feasibility (RMF) mathematics:

- `src/dnadesign/opal/docs/plugins/objectives/response-magnitude-feasibility.md`
- `src/dnadesign/opal/src/objectives/response_magnitude_feasibility_math.py`
- `src/dnadesign/opal/src/objectives/response_magnitude_feasibility_v1.py`

Reader response-window contract:

- `reader/docs/lib/plate_reader/response_window.md`
- Reader bundle schema `reader.response_window.bundle.v4`

### Package Layout

- `core/`: stress target views, SFXI source provenance, policy, path, and evidence contracts.
- `evaluation/`: metric behavior, target margins, uncertainty, fixed model
  comparisons, repeated measurements, and greedy-support evidence.
- `runtime/`: OPAL loading, Reader-bundle verification, orchestration,
  publication, and manifests.
- `reporting/`: declarative plot contracts, plot writers, report sections, and
  Marimo generation.
- `config/reader_response_window.yaml`: study-owned Reader service request.

Reader trajectory loaders, event inference, and time reducers do not exist in
the study package.

### Metric identities

Reader's response-window handoff is
`[r00, r10, r01, r11, b00, b10, b01, b11]`. Here `r_i` is reduced
`log2(YFP/CFP)` response and `b_i` is same-state pDual-10-relative
`log2(YFP/OD600)` fluorescence. These values are not the SFXI vec8 fields and
must not use `sfxi_vec8` names. The primary reduction is the 6-12-hour
post-event geometric log mean. Reader owns that reduction and its uncertainty;
the study owns repeat aggregation and promotion of response-window observed Y;
OPAL applies RMF and owns campaign scoring.

### Evidence Findings

Canonical SFXI recomputes exactly, but its persisted top-six results remain
effect-dominated:

- 18 target-view slots collapse to 11 unique sequences;
- 2 candidates appear in all three target-view lists;
- weakest median top-six logic fidelity is 0.258;
- mean pairwise target-view score correlation is 0.968.

The Reader bundle contains 8 experiments, 5 reductions, 295 design/reduction
rows, 147,500 joint bootstrap rows, and 12 repeated design IDs. The primary
reduction is `event_logmean_6_12h_post`.

The leading screened model challenger is PLS4 over the primary eight-component
summary. Its weakest response-separation Spearman is 0.15 and weakest
feasibility Spearman is 0.45 across active views. This defines a directional
experiment, not a calibrated success probability.

Retrospective enrichment is strongest for ciprofloxacin, intermediate for AND,
and weakest for ethanol, but all exact 95% intervals include 0.5. The configured
selection mechanism under review is greedy top-six per view. It is not promoted
or authorized for synthesis; the metastudy records its risk and does not derive
slot allocations from these uncertain intervals.

### Plot Surface

Four ordered primary plots carry the decision narrative:

- a measured-example check shows how the same SpyP and sulAp summaries change
  when ethanol, ciprofloxacin, and AND masks reassign ON and OFF states, without
  treating either promoter as a required paradigm;
- response-window stability tests whether the promoted reduction preserves the
  ordering of the three RMF components;
- the grouped label-model screen tests whether X preserves RMF ordering in held-out
  Reader experiments;
- the greedy-support interval tests whether predicted leaders enrich held-out
  measurements beyond the experiment median and bounds the claim made by a
  prospective greedy round.

Metric diagnostics explain behavior without deciding promotion:

- canonical SFXI component dominance, target residuals, Pareto views, score
  correlations, and candidate support;
- Reader event intervals, uncertainty sources, and response-reduction
  sensitivity;
- SFXI vec8 model validation, observed constraint coverage, repeated-design
  variation, and retrospective enrichment.

Screen appendices retain the policy guardrail matrix and complete policy,
selected-profile, and overlap sweeps.
Every plot declares one premise, a concise title without terminal
punctuation, decision value, rationale, alt text, non-claim boundary, tier, and
source table. Plot files use a white canvas independent of notebook theme.

The generated Marimo review uses the medium-width layout, one responsive
tier-and-figure control row, and one image viewport constrained to the notebook
column. Compact matrix plots use square cells. The repeated-design matrix groups
`r00`, `r10`, `r01`, and `r11` under `log2(YFP/CFP)` response and `b00`, `b10`,
`b01`, and `b11` under pDual-10-relative
`log2(YFP/OD600)` fluorescence, while naming each stress condition below the
group header. The row-dense policy guardrail remains rectangular because square
cells would create a needlessly tall appendix figure.

### Run

From `dnadesign/`:

```bash
uv run python -m \
  dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy \
  --reader-bundle ../reader/outputs/reviews/stress_response_window/latest \
  --candidate-bindings src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/promoter_candidate_bindings/latest \
  --overwrite \
  --json
```

The command stages output atomically, verifies generated artifacts, and replaces
the destination only after the complete run succeeds.

### Promotion Boundary

No objective or model is promoted until the study has one repeated-experiment
aggregation rule, one typed OPAL label contract, grouped evidence for all active
selection views, and an explicit run decision. The nearest-12-hour vec8 remains
immutable provenance after any new label is promoted.

The metastudy can reject unsupported choices. A biological hill climb exists
only after a prospective selected set is built, measured, and compared with its
declared baseline and controls.
