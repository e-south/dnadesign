---
id: stress-ethanol-cipro-growth-opal-response-metastudy
title: Response assay and objective comparison
owner: dnadesign-maintainers
status: source_evidence
last_verified: 2026-07-21
audience:
  - scientist
  - maintainer
  - agent
---

## Response assay and objective comparison

**Status:** frozen comparative evidence
**Owner:** `stress_ethanol_cipro_growth` study
**Last verified:** 2026-07-21
**Implementation:** `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy`
**Generated evidence:** `src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/response_metastudy/latest`

### Premise

The metastudy separates label truth from model support. The response-window
label contract passed its study-owned promotion gate, while held-out ordering
remained weak. A separate explicit study decision authorized one frozen MSRB
greedy probe without changing `model_support_ready=false`. Technical
executability, predictive support, and synthesis authorization are distinct
decisions.

### Scope

The metastudy addresses three distinct questions:

1. Which response window and label-source rules produce an auditable
   candidate-level phenotype?
2. What do grouped model checks say about prediction and ranking support?
3. What did the historical SFXI and RMF comparisons reveal about objective
   behavior before MSRB became the executable selector?

It is read-only with respect to Reader records, OPAL labels, campaign configs,
ledgers, and synthesis handoffs.

### Ownership

- Reader owns event resolution, trajectory reduction, within-experiment
  replicate aggregation, joint bootstrap draws, reference subtraction, and
  assay review visuals.
- The stress study owns target masks, Reader-to-candidate joins, label
  representation comparisons, repeated-candidate label-source decisions,
  grouped model evaluation, and promotion policy.
- OPAL owns canonical SFXI, Response-Magnitude Feasibility (RMF), and MSRB
  mathematics, plus model fitting, candidate scoring, selection, and ledgers
  after promotion.

The metastudy consumes Reader's public bundle. It does not import Reader or
maintain duplicate trajectory math.

### Campaign, target-view, and source boundaries

The raw response-window prediction run that originated in
`secg_rmf_greedy` is frozen under the study-owned
`workbench/source_evidence/opal_response_window_round0/` shelf. The original
slug is provenance; this is comparator evidence, not an executable campaign.
The metastudy loads that fixed source contract and
derives exactly three immutable
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
choice for repeated designs explicit, does not read an SFXI source CSV, and has
no label-truth or calibration-cohort role. Cross-experiment evidence, explicit
label sources, and approval live in the study-level
`response_window_observations/` package. Campaign scales use every exact,
study-bound, non-reference primary Reader candidate-experiment unit rather than
this screen selection.

The response-metastudy publication schema is
`stress_ethanol_cipro_growth.response_metastudy.v12`.

### Evidence Flow

1. Verify `reader.response_window.bundle.v5`, all record contracts, source
   provenance, artifact digests, and row counts.
2. Verify the response-owned 35-row screen selection against the Reader bundle,
   resolve each design alias through the study binding artifact, and join by
   exact Reader experiment and design identity.
3. Evaluate response and fluorescence requirement stability across seven
   Reader-owned event-relative reductions. Compare pDual-10 replicate spread,
   cross-experiment anchor drift, SpyP and sulAp response separation, OD context,
   event sensitivity, repeat agreement, censoring support, and the same fixed
   model screen for every reduction. Model performance is diagnostic and does
   not choose the response window.
4. Apply ethanol, ciprofloxacin, AND, and OR pressure-test masks to raw Reader
   state summaries and joint bootstrap draws.
5. Compare the exact configured campaign RF separately from the fixed mean,
   robust-target RF, fold-fitted PCA-ridge, and PLS challengers with complete
   Reader experiments held out.
6. Measure repeated-design agreement, retrospective enrichment, and the risk of
   a prespecified coordinated six-slot policy.
7. Verify the three immutable round-0 SFXI source ledgers, shared 35-row label
   pool, candidate IDs, and equivalent predictor surfaces. Recompute persisted
   SFXI scores through the public OPAL API, decompose each measured score into
   logic fidelity and scaled effect, and test the result after deleting each
   source experiment and restricting the comparison to ES designs.
8. Publish typed tables, a manifest-backed plot catalog, a report, and one
    Marimo review notebook.

The minimal review path is Reader phenotype -> target pattern -> objective
decomposition -> grouped model evidence -> allocation diagnostics. SFXI, RMF,
and MSRB remain distinct objectives with distinct claim boundaries. Parameter
sweeps, overlap screens, and alternative reductions remain sensitivity or
comparator evidence rather than production selectors.

### Canonical Sources

Canonical SFXI mathematics:

- `src/dnadesign/opal/docs/plugins/objectives/sfxi.md`
- `src/dnadesign/opal/src/objectives/sfxi_math.py`
- `src/dnadesign/opal/src/objectives/sfxi_v1.py`

Response-Magnitude Feasibility (RMF) mathematics:

- `src/dnadesign/opal/docs/plugins/objectives/response-magnitude-feasibility.md`
- `src/dnadesign/opal/src/objectives/response_magnitude_feasibility_math.py`
- `src/dnadesign/opal/src/objectives/response_magnitude_feasibility_v1.py`

Multistate Response Behavior objective and study evidence:

- `src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md`
- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/config/multistate_response_behavior_shadow_v1.yaml`
- `docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior.md`

Reader response-window contract:

- `reader/docs/lib/plate_reader/response_window.md`
- Reader bundle schema `reader.response_window.bundle.v5`

### Package Layout

- `core/`: stress target views, SFXI source provenance, policy, path, and evidence contracts.
- `evaluation/`: metric behavior, target margins, uncertainty, fixed model
  comparisons, repeated measurements, and greedy-support evidence.
- `runtime/`: OPAL loading, Reader-bundle verification, orchestration,
  publication, and manifests.
- `reporting/`: declarative plot contracts, plot writers, report sections, and
  Marimo generation.
- `response_window_observations/config/reader_response_window.yaml`: study-owned
  Reader service request shared by evidence and decision workflows.

Reader trajectory loaders, event inference, and time reducers do not exist in
the study package.

### Metric identities

Reader's response-window handoff is
`[r00, r10, r01, r11, b00, b10, b01, b11]`. Here `r_i` is reduced
`log2(YFP/CFP)` response and `b_i` is same-state pDual-10-relative
`log2(YFP/OD600)` fluorescence. These values are not the SFXI vec8 fields and
must not use `sfxi_vec8` names. The primary reduction is the 4-8-hour
post-event geometric log mean. Reader owns that reduction and its uncertainty;
the study owns repeated-candidate label sources and promotion of response-window
observed Y; OPAL applies the configured objective and owns campaign scoring.

### Evidence Findings

Canonical SFXI recomputes exactly on the 35 historical observed labels. Across
the ethanol-associated, ciprofloxacin-associated, and combined-state-only
views, the rank correlations between SFXI and scaled effect are 0.967, 0.920,
and 0.955. The corresponding correlations with logic fidelity are -0.209,
-0.030, and -0.170. Deleting each source experiment in turn or restricting the
analysis to the 23 ES designs preserves the conclusion. These are
corpus-sensitivity checks, not cross-validation or evidence that SFXI is
universally effect-dominated.

The persisted predicted top-six results show the related allocation pattern:

- 18 target-view slots collapse to 11 unique sequences;
- 2 candidates appear in all three target-view lists;
- weakest median top-six logic fidelity is 0.258;
- mean pairwise target-view score correlation is 0.968.

The Reader bundle contains 8 experiments, 7 reductions, 413 design/reduction
rows, 206,500 joint bootstrap rows, and 12 repeated design IDs. The primary
reduction is `event_logmean_4_8h_post`.

The strongest descriptive fixed challenger in this snapshot is PLS4 over the primary eight-component
summary. Its weakest response-separation and feasibility Spearman values are
both 0.45 across active views. This defines a directional
experiment, not a calibrated success probability or a campaign-model change.
The exact configured campaign RF is reported on its own row and is the only
model that can satisfy the campaign-model support gate.

Retrospective enrichment is strongest for ciprofloxacin, intermediate for AND,
and weakest for ethanol, but all exact 95% intervals include 0.5. Round 0 used
one shared RF and deterministic round-robin allocation to assign six
sequence-unique slots per view. The preferred lists contained one overlap, so
the AND view advanced once to produce 18 unique sequences. This operational
completion does not promote the model or authorize synthesis; the metastudy
records predictive risk and does not derive slot allocations from uncertain
retrospective intervals.

### Plot surface

The Marimo review has one image viewport and two controls: **Review section**
and **Figure**. The sections follow the evidence path rather than the directory
layout:

1. **Assay and labels** opens on the 4–8-hour response-window comparison, then
   exposes event timing and repeated-design agreement.
2. **Model support** contains grouped prediction checks and keeps weak
   prospective ordering visible.
3. **RMF comparator** retains thresholded requirements, uncertainty, and
   retrospective greedy evidence as comparator material.
4. **SFXI comparator** opens with the measured-label decomposition, then
   retains predicted setpoint, score-coupling, overlap, and policy screens under
   the distinct SFXI phenotype contract.

The active `secg_msrb_greedy` campaign is reviewed in its own OPAL notebook.
The metastudy does not reproduce active campaign state or use comparator plots
as selection authority.

Every plot declares one premise, a concise title without terminal punctuation,
decision value, rationale, alt text, non-claim boundary, review section,
storage tier, and source table. Plot files use a white canvas independent of
notebook theme. Compact matrix plots use square cells. The repeated-design matrix groups
`r00`, `r10`, `r01`, and `r11` under `log2(YFP/CFP)` response and `b00`, `b10`,
`b01`, and `b11` under pDual-10-relative
`log2(YFP/OD600)` fluorescence, while naming each stress condition below the
group header. The row-dense policy guardrail remains rectangular because square
cells would create a needlessly tall comparator figure.

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

The label-truth gate is complete: the checked-in source policy selects eight
reviewed repeat sources, excludes four unresolved repeated candidates, and the
typed publication verifies 27 exact labels plus eight measured-candidate
exclusions. That result does not satisfy the independent model-support gate.
`model_support_ready` remains false. The sole executable stress campaign is the
MSRB round-0 prospective learning probe; the RMF run is frozen comparator
evidence. Explicit probe authorization did not promote model support. The study
later accepted the exact assay-batch-1 synthesis handoff as a separate physical
decision; that acceptance does not change the model-support result. The
nearest-12-hour SFXI vec8 remains immutable provenance and is not an RMF label.

The metastudy can reject unsupported choices. A biological hill climb exists
only after a prospective selected set is built, measured, and compared with its
declared baseline and controls.
