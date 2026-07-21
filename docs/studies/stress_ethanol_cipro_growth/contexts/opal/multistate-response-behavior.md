---
id: stress-ethanol-cipro-growth-multistate-response-behavior
title: Multistate Response Behavior in the stress-promoter study
short_name: MSRB
objective_id: multistate_response_behavior_v1
y_space: reader_response_window_vector_v1
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-20
first_hop: ../../../../../src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md
audience:
  - scientist
  - maintainer
  - operator
  - agent
---

## Multistate Response Behavior in the stress-promoter study

This campaign uses Multistate Response Behavior (MSRB) to rank promoter
phenotypes toward three binary stress-response patterns. Here, a phenotype is
the ordered set of values that describes one promoter across the four measured
conditions. Reader reduces each experiment to eight such values; the stress
study resolves candidate identity and publishes reviewed labels; OPAL predicts
the complete phenotype and scores the same prediction under each target
pattern.

The [generic MSRB definition](../../../../../src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md)
defines the reusable phenotype, equations, diagnostics, and limits. This page
records the assay-specific reduction, label policy, target masks, soft-min
scale, model, allocation rule, and evidence boundaries.

The objective identifier is `multistate_response_behavior_v1`. Its selected
score channel is `behavior_score`, written $S_{\mathrm{MSRB}}$ and maximized.
MSRB is the campaign's only ranking objective.

The model, objective, and allocation rule are fixed before the selected batch
is measured. The run therefore tests a declared greedy ranking policy rather
than revising that policy after seeing the results. Existing evidence supports
the calculation and its alignment with the stated binary ON/OFF preference,
but not reliable predictive ordering, prospective enrichment, or hill-climb
efficacy. Selection does not authorize synthesis; the study separately accepted
the exact `SECG-019` through `SECG-036` physical handoff.

### Study binding

| Element | Study choice |
| --- | --- |
| Measured states | `00`: no stress; `10`: ethanol; `01`: ciprofloxacin; `11`: both stresses |
| Primary response window | 4–8 hours after intervention |
| Response coordinate, $r_i$ | Reduced base-2 log ratio of yellow fluorescent protein (YFP) to cyan fluorescent protein (CFP), measured from the same plasmid in each well |
| Signal coordinate, $b_i$ | Reduced base-2 log ratio of YFP to optical density at 600 nm (OD600), relative to pDual-10 measured in the same state on the same plate |
| Signal reference | pDual-10, measured in the candidate's state on the same plate |
| Phenotype | `[r00, r10, r01, r11, b00, b10, b01, b11]` |
| Target views | Ethanol-associated, ciprofloxacin-associated, and combined-state-only |
| Soft-min scale | One shared $\tau\approx0.31$ log2 |
| Model target | Complete eight-value phenotype |
| Selector | Greedy `top_n`, six per view |
| Allocation | Round-robin next-best-unallocated, 18 sequence-unique candidates |

The exact binding is recorded in
[`protocol.yaml`](../../../../../src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/protocol.yaml).

### End-to-end evidence path

```text
OD600, YFP, and CFP well trajectories
  -> event-relative 4-8 h reduction for each well
  -> within-experiment replicate medians
  -> same-state pDual-10 signal subtraction
  -> one Reader experiment-level response-window phenotype
  -> identity-verified Reader alias-to-candidate binding
  -> study review of repeated experiments
  -> one candidate-level observed phenotype
  -> verified candidate-level OPAL label
  -> sequence representation X
  -> predicted response-window phenotype Y_hat_RW
  -> target-view MSRB score
  -> top-six ranking per view
  -> sequence-unique 18-candidate allocation
  -> prospective measurement
```

Each stage preserves the prior stage's artifact rather than recalculating its
measurements or identities.

### 1. Reader reduces each experiment

Reader starts from annotated OD600, YFP, and CFP trajectories. It resolves the
intervention interval, expresses time relative to that event, and averages
each log2 ratio from 4 to 8 hours after intervention. On the original ratio
scale, this is equivalent to a geometric time mean.

For each well $w$, Reader calculates

$$
R_w = \frac{1}{4\ \mathrm{h}}
\int_{4\ \mathrm{h}}^{8\ \mathrm{h}}
\log_2\left(\frac{\mathrm{YFP}_w(t)}{\mathrm{CFP}_w(t)}\right)dt,
$$

$$
F_w = \frac{1}{4\ \mathrm{h}}
\int_{4\ \mathrm{h}}^{8\ \mathrm{h}}
\log_2\left(\frac{\mathrm{YFP}_w(t)}{\mathrm{OD600}_w(t)}\right)dt.
$$

In $R_w$, YFP and CFP are the two reporters on the same assayed plasmid. The
ratio is calculated within each well; it is not a comparison between separate
constructs.

Each well is reduced before wells are combined. Pooling raw time points across
wells would give densely sampled wells more influence and would erase the
replicate structure.

In the assay-development plates, the 4–8-hour window retained much of the SpyP
and sulAp separation seen at 6–12 hours while reducing late-plate OD
accumulation, overflow exposure, and plate-age effects. The window was chosen
with the available corpus and is now fixed for the prospective probe. It is
not independent validation of the window. The [response
metastudy](response-metastudy.md) records the equal-footing window comparison.

Alternative windows, event bounds, area-under-the-curve summaries, and delta
summaries remain sensitivity evidence. They do not replace the primary
reduction. Reader owns the full contract in
`reader/docs/lib/plate_reader/response_window.md` in the sibling Reader
repository.

### 2. Reader summarizes wells within an experiment

For state $i$, Reader combines replicate wells from the same experiment with
medians:

$$
r_i = \operatorname{median}_{w \in \mathrm{design},i} R_w,
$$

$$
b_i =
\operatorname{median}_{w \in \mathrm{design},i} F_w
{}-
\operatorname{median}_{w \in \mathrm{pDual\text{-}10},i} F_w.
$$

This produces one estimate per experiment, design, and state. The pDual-10
reference is measured in the same state on the same plate as the candidate. A
candidate is never compared with pDual-10 from another plate or state.

pDual-10 is the study's condition-matched promoter reference. The [dataset
registry](../../record/datasets.yaml) records its construct and source lineage.
The calculation uses measured pDual-10 output and does not assume that the
reference is condition-invariant.

Reader publishes three evidence layers beside the central estimates:

- a well-resampling bootstrap shows sensitivity to which wells were measured;
- event-time bounds rerun the reduction at the earliest and latest recorded
  intervention times and show the largest change from the central estimate;
  and
- censor provenance identifies values bounded by instrument overflow or a
  declared policy.

These layers are not averaged into the central phenotype or appended as a
ninth value. Well resampling creates no new observations and does not estimate
variation across future experiments. Within a state, Reader keeps the response
and design-signal contributions paired during resampling. When pDual-10 is
compared with itself, the same sampled reference wells appear on both sides,
so every $b_i$ draw is exactly zero rather than carrying artificial
self-comparison noise.

Together, the four $r_i$ values and four $b_i$ values form one
experiment-level phenotype:

$$
Y_{\mathrm{RW}}=
[r_{00},r_{10},r_{01},r_{11},b_{00},b_{10},b_{01},b_{11}].
$$

This is the eight-component object that the study reviews when a candidate has
one or more experiments. Section 4 provides a compact coordinate reference.

### 3. The study establishes candidate-level labels

One Reader experiment is one evidence unit, regardless of how many valid wells
it contains. When a candidate appears in more than one experiment, the study
does not pool all wells, infer authority from a timestamp, or average
discordant experiments automatically.

The study applies an explicit disposition:

- a singleton experiment contributes its uncensored phenotype;
- an accepted repeated candidate contributes one reviewed Reader experiment;
- a discordant or unresolved candidate is excluded pending remeasurement; and
- the selected label-source experiment must have eight uncensored primary
  components.

A bounded component in a nonselected experiment remains repeat evidence and
does not block promotion from a reviewed uncensored source. Here, `uncensored`
means only that none of the selected source's eight components is a bound. It
does not mean error-free, repeat-stable, population-average, or representative
of a future experiment.

Every contributing experiment, exclusion, Reader manifest digest, candidate
binding, and uncertainty source remains recorded. Cross-experiment differences
remain repeat diagnostics even when one reviewed source supplies the training
label.

The study-owned, digest-matched candidate binding requires each Reader alias to
resolve to one candidate and a matching sequence digest. Missing, duplicated,
fuzzy, prefix-based, or sequence-mismatched joins fail. A Reader alias does not
become candidate authority by itself.

The current label set contains 27 candidate phenotypes from uncensored selected
sources. OPAL verifies its `opal.observed_label_promotion.v1` receipt and
artifact digests before reading the labels. Across campaign rounds, the ledger
retains every label event; the configured `latest_only` policy chooses the
latest uncensored event per candidate for cumulative training. That round-level
rule is separate from study review of repeated Reader experiments within one
event.

### 4. Coordinate reference for the eight-value phenotype

Generic MSRB consumes `2K` ordered values for any fixed `K >= 2`. This assay
has four states, so Reader publishes

$$
Y_{\mathrm{RW}}=
[r_{00},r_{10},r_{01},r_{11},b_{00},b_{10},b_{01},b_{11}].
$$

| Coordinate | Meaning |
| --- | --- |
| $r_i$ | Median well-level 4–8-hour reduced log2(YFP/CFP) response from the same plasmid in state $i$ |
| $b_i$ | Median design log2(YFP/OD600) minus median pDual-10 log2(YFP/OD600), measured in the same state on the same plate |
| $b_i=0$ | Equal reduced signal to same-state, same-plate pDual-10 |
| $b_i=1$ | Twofold higher reduced signal than same-state, same-plate pDual-10 |
| $b_i=-1$ | Twofold lower reduced signal than same-state, same-plate pDual-10 |

The phenotype does not contain a target mask, objective score, uncertainty
value, candidate rank, or synthesis decision. Its meaning is objective-neutral.

### 5. OPAL predicts the complete phenotype

The campaign trains one multi-output random forest:

$$
X_{\mathrm{sequence}}\longrightarrow\widehat{Y}_{\mathrm{RW}}.
$$

$X_{\mathrm{sequence}}$ is the declared 8,192-component sequence
representation. The training target is the complete observed eight-value
phenotype. The model does not fit one scalar per target view and does not
predict MSRB directly.

OPAL scores the same prediction under each target mask:

$$
X_{\mathrm{sequence}}
\longrightarrow \widehat{Y}_{\mathrm{RW}}
\longrightarrow S_{\mathrm{MSRB},v}
\longrightarrow \operatorname{topN}_v.
$$

Changing the target mask changes the interpretation and ranking of a
prediction; it does not retrain the model. This keeps component-level
prediction errors observable and avoids three disconnected scalar-model
lineages.

The 100-tree random forest is a fixed prospective baseline, not a claim that
random forests are optimal. It is trained on 27 labels from six selected source
experiments against an 8,192-component representation. Aggregate out-of-bag
$R^2$ is approximately `0.067`; this internal leave-some-training-rows-out
diagnostic explains little of the observed variation. Retrospective
experiment-held-out rank support is also weak. The prospective run tests these
frozen predictions rather than assuming they are accurate.

On fixed grouped validation, median held-out Spearman rank correlations were
as follows. A value near `1` preserves ordering, while a value near `0` carries
little monotonic ordering information. The table includes the frozen
[Response-Magnitude Feasibility (RMF)](response-magnitude-feasibility.md)
comparator.

| Objective | Combined-state-only | Ciprofloxacin-associated | Ethanol-associated |
| --- | ---: | ---: | ---: |
| MSRB | `0.257` | `-0.200` | `0.086` |
| RMF | `0.143` | `-0.071` | `-0.086` |

Neither row supports reliable ranking. MSRB was chosen for semantic alignment,
not because this table establishes predictive superiority.

### 6. The study defines target views and score scale

The generic objective defines the three behavior families, equal total family
weights, and the soft minimum. This study supplies the ordered states, masks,
and one shared soft-min scale.

#### Target views

| View | Plain interpretation | Mask `[00, 10, 01, 11]` | Intended-ON states |
| --- | --- | --- | --- |
| Ethanol | Ethanol-associated pattern | `[0, 1, 0, 1]` | Ethanol; both stresses |
| Ciprofloxacin | Ciprofloxacin-associated pattern | `[0, 0, 1, 1]` | Ciprofloxacin; both stresses |
| AND | Combined-state-only pattern | `[0, 0, 0, 1]` | Both stresses only |

The ethanol-associated view (`[0, 1, 0, 1]`) compares every ethanol-present
response with every ethanol-absent response:

$$
r_{10}-r_{00},\quad r_{10}-r_{01},\quad
r_{11}-r_{00},\quad r_{11}-r_{01}.
$$

This is global ordering by target membership. It is not limited to the matched
conditional effects $r_{10}-r_{00}$ and $r_{11}-r_{01}$. The
ciprofloxacin-associated view (`[0, 0, 1, 1]`) uses the analogous all-pairs
ordering. The combined-state-only view (`[0, 0, 0, 1]`) asks for $r_{11}$ to
exceed $r_{00}$, $r_{10}$, and $r_{01}$; it does not calculate biochemical
interaction or the superadditive contrast
$r_{11}-r_{10}-r_{01}+r_{00}$.

For any one mask, the predicted phenotype becomes three behavior families:

| Family | Values supplied to MSRB | Favorable direction |
| --- | --- | --- |
| Response ordering | Every intended-ON $r_i$ minus every intended-OFF $r_j$ | Larger ON-over-OFF differences |
| Intended-ON signal | Each intended-ON $b_i$ | Brighter than the same-state pDual-10 reference |
| Intended-OFF suppression | The negative of each intended-OFF $b_j$ | Dimmer than the same-state pDual-10 reference |

Each family receives one third of the starting weight, regardless of how many
state comparisons it contains. The shared soft minimum then combines all
values into `behavior_score`. The [generic objective
definition](../../../../../src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md)
gives the complete equations and compensation examples.

#### What the soft-min scale changes

All scored values are log2 differences: ON-minus-OFF response,
intended-ON signal relative to pDual-10, and the negative of intended-OFF
signal relative to pDual-10. The campaign uses one shared scale of approximately
$0.31$ log2.

The scale controls how readily strong behavior compensates for weak behavior.
It is not a biological threshold, a minimum detectable effect, or a
candidate-specific uncertainty penalty. Using one shared value avoids an
additional preference about how much a response change is worth relative to a
signal change.

#### Why the study uses approximately 0.31 log2

The study derived the value from within-experiment well-resampling evidence:

1. Start with 41 candidate-by-experiment units whose primary eight values are
   uncensored. They cover 32 candidates and eight Reader experiments; six
   units containing at least one bounded primary value are excluded.
2. Use 500 Reader well-resampling draws for each unit while preserving the
   paired response and signal calculations.
3. Calculate one standard deviation for each of six distinct response
   state-pair differences and four state-specific pDual-10-relative signal
   values.
4. Pool those standard deviations and take their 90th percentile.

| Resampling summary | Count | 90th percentile SD (log2) |
| --- | ---: | ---: |
| Response state-pair differences | 246 | `0.308` |
| pDual-10-relative state values | 164 | `0.313` |
| Pooled values used for $\tau$ | 410 | `0.311` |

The response and signal summaries differ by less than `0.005 log2` at the
chosen percentile. A shared value therefore removes a second scale without
hiding a meaningful difference at that percentile.

#### What the resampling does not establish

The 90th percentile is an assay-development convention, not a biological
constant. It places the compensation scale near the upper end of observed
within-experiment resampling variation without letting a few extreme values
set it. The pooled median is `0.136 log2`, the 95th percentile is `0.361
log2`, and the maximum is `1.130 log2`. The available data do not uniquely
select the 90th percentile.

The 410 values are correlated summaries from 41 units, not 410 independent
experiments. They describe sensitivity to resampling measured wells. They do
not estimate instrument resolution, between-experiment reproducibility, model
uncertainty, or a limit of detection. The rule was developed on the available
assay corpus rather than independently validated on a new corpus.

#### Sensitivity and reproducibility

For unit intuition, $2^{0.31}\approx1.24$. For a response difference, this is
a ratio of YFP/CFP ratios; for a signal value, it is a design-to-pDual-10
YFP/OD600 ratio. It is not a desired 1.24-fold biological effect.

Under the current masks, the behavior score can lie at most `0.34–0.77 log2`
above its hard bottleneck, depending on the limiting coordinate's family. An
implicit scale of `1 log2` would raise the largest bound from `0.77` to `2.48
log2`; omitting the field would therefore hide a more permissive rule rather
than remove a choice.

Across 13 prespecified scale and source-cohort checks, pool-wide ranks remained
broadly similar to the selected convention (minimum Spearman correlation
`0.955`), but raw top-six overlap fell to `3/6` in the most sensitive case.
Broad ordering was fairly stable; exact greedy nominations remained
scale-sensitive. The value must therefore be fixed before allocation.

Well-resampling bootstrap draws are not inserted into candidate scores or used
as candidate-specific penalties. The protocol stores the exact value and recipe.
The executable value is
$\tau=0.31063783855250376$ log2; the additional digits preserve the recorded
calculation and do not imply matching biological precision.
The generated
`workbench/outputs/multistate_response_behavior_shadow/latest/normalization.json`
record and its supporting tables preserve the calculation; the activation
receipt binds their verified bundle.

Scores are comparable only within one view under the same state order, mask,
scale, and objective version. An ethanol-associated score and a
combined-state-only score are not interchangeable measurements.

### 7. OPAL ranks and allocates candidates

For each view, `top_n` ranks candidates by decreasing `behavior_score` and
nominates six. MSRB is the objective; `top_n` is the selector. The policy is
deterministic greedy exploitation, not Bayesian optimization or
uncertainty-aware acquisition.

OPAL then allocates one sequence at a time in the declared view order:
ethanol, ciprofloxacin, and AND. If a sequence was claimed by an earlier view,
the current view advances to its next-best unallocated candidate. The batch
must contain six candidates per view and 18 unique sequences or the run fails.
Sequence uniqueness prevents exact duplicates; it is not a general diversity
criterion.

Every candidate is measured in all four states, so it supplies evidence for
all three views, not only the view that nominated it.

### Reading the campaign landscape

The campaign notebook shows the three family scores:

- farther right means stronger response ordering, $S_R$;
- farther up means stronger intended-ON signal, $S_{ON}$; and
- larger labeled values on the colorbar or z-axis mean stronger intended-OFF suppression,
  $S_{OFF}$.

The hue is only a display encoding of the third family. It does not represent
YFP color or a separate biological quantity. A candidate that improves on all
three axes must receive a higher MSRB score. A point that is only higher and
farther right can rank below another point when its OFF suppression is worse.

Selected predictions need not be the visually highest observed points. SpyP,
sulAp, pDual-10, and other observed labels are assay evidence, not unmeasured
DenseGen candidates eligible for selection. The selector ranks all three
families rather than either plotted axis alone, and sequence-unique allocation
can advance a later view to its next-best unallocated sequence.

The selected-candidate decomposition gives the defensible explanation of one
rank: every state-level clearance, the three family scores, the hard
bottleneck, and the final smooth score. Zero is a reference direction, not a
biological acceptance threshold.

The current prediction surface is narrow. No predicted candidate meets every
reference direction for the ethanol-associated or combined-state-only views;
about 19.4% do for the ciprofloxacin-associated view. Some
combined-state-only predictions have a positive behavior score while every
candidate still has a negative hard bottleneck. These are least-bad predicted
compromises, not predicted conforming programs.

Predicted ON-signal and OFF-suppression family scores are strongly
anticorrelated: Spearman correlations are approximately `-0.924`, `-0.948`,
and `-0.986` for the ethanol-associated, ciprofloxacin-associated, and
combined-state-only views. After standardization, the third principal
direction explains only about 0.1–0.2% of family-score variance. The fitted
model mainly represents a global-expression tradeoff in which brighter ON
states accompany brighter OFF states. This does not make the three
mathematical families duplicates. No score can select a phenotype combination
that the predictor does not place in its prediction surface.

The per-view counts, correlations, and principal-component variance ratios are
generated from the fixed prediction-score table and recorded in the generated
`tables/prediction_surface_diagnostics.parquet` evidence table.

The OFF-suppression display uses one symmetric linear extent for all three
views, fixed from the absolute 99th percentile of the round-0 prediction pool.
Values outside the extent keep their exact scores and remain plotted, but
their colors saturate at the colorbar endpoints. The caption reports the
saturated count. This view-independent display rule changes neither the
objective nor candidate rank.

### Evidence supporting the present probe

MSRB was chosen because the target is binary and directional, every favorable
change should affect rank, and no biological pass threshold is justified. Its
declared preferences remain visible: binary target membership, equal total
weight for three behavior families, one compensation scale, and bounded
tradeoffs.

RMF is better suited to validated engineering thresholds that must all be met.
Its hard minimum prevents compensation, but a nonlimiting improvement is
invisible and one noisy extreme can control rank. The fixed zero-boundary RMF
run remains comparator evidence rather than a second executable selector.

[Setpoint Fidelity × Intensity
(SFXI)](../../../../../src/dnadesign/opal/docs/plugins/objectives/sfxi.md) uses a
distinct Reader vec8 and does not directly score target-OFF signal. Vector
target similarity requires absolute setpoints and penalizes overshoot. A
single vector channel omits two behavior families. Scalar identity and
[SPOP](../../../../../src/dnadesign/opal/docs/plugins/objectives/spop.md) belong
to different Y contracts.

SpyP ranks near the top of observed ethanol-associated examples but retains an
OFF-suppression failure. SulAp ranks near the top of observed
ciprofloxacin-associated examples. These are limited face-validity checks, not
score thresholds or DenseGen architecture matches. The study has no equivalent
positive combined-state-only control.

The objective tests establish directional monotonicity, Pareto dominance
(improving every coordinate cannot lower rank), unchanged results after a
consistent reordering of states, equal family standing, bounded compensation,
numerical stability for extreme finite inputs, and state-panel pressure from
`K=2` through `K=16`. Study review additionally covered the active masks, Reader bootstrap
and event-time sensitivity, repeated experiments, censoring, scale
sensitivity, and sequence-deduplicated allocation.

These checks support the equations and implementation. They do not repair the
weak sequence-to-phenotype predictor. Greedy Top-K searches the largest
predicted scores among 154,785 candidates, where the largest model errors may
also occur. The campaign has neither a tested estimate of prediction
uncertainty nor a safeguard for sequences unlike the training examples. Under
the same prediction matrix, MSRB and RMF differ on 3 of 18 allocated
sequences. Their 15-of-18 overlap does not establish either objective's
prospective efficacy.

### Uncertainty and censoring

Uncertainty is not a ninth phenotype component and does not enter the current
selector.

| Evidence | Meaning | Current role |
| --- | --- | --- |
| Well-resampling bootstrap | Sensitivity to which within-experiment wells were measured; not future-experiment variation | Scale derivation and rank-sensitivity review |
| Event-time bounds | Sensitivity to uncertainty in intervention timing | Separate envelope review |
| Repeat disagreement | Dependence on which experiment measured one candidate | Study adjudication and exclusion |
| Censor provenance | Exact, lower-bounded, or upper-bounded assay value | Exact-label gate |
| Model uncertainty | Epistemic uncertainty in $X\rightarrow Y$ | Not calibrated or used by the selector |

Reader bootstrap variation must not be relabeled as model uncertainty. Random-
forest tree dispersion is not a calibrated posterior. An uncertainty-aware
acquisition policy would require a separate validated uncertainty contract.
Between-experiment disagreement remains the larger unresolved measurement
variation and is handled by study review rather than hidden in MSRB.

### Required observability

The campaign needs five evidence surfaces:

1. Reader evidence for the selected time window, trajectories, replicate
   support, censoring, and growth context.
2. The three family scores, hard bottleneck, limiting coordinate, and
   direction-met status for every selected candidate.
3. One pool-wide landscape that distinguishes predictions, observations,
   selections, and allocation replacements.
4. Experiment-held-out raw-$Y$ and objective-ranking validation with sample
   counts and undefined folds visible.
5. After measurement, one frozen table containing predicted and observed
   $Y_{\mathrm{RW}}$, family scores, MSRB, raw rank, allocated view and rank,
   and the declared comparison.

BaseRender sequence evidence and artifact digests support identity and
provenance; they do not add another scored behavior. Round-over-round plots are
appropriate after at least two measured batches and should show predictions
made before each measured batch against a prespecified baseline. Cumulative
best score alone tends to rise as more candidates are measured and is not
evidence of learning.

### Prospective evaluation

The first measured batch evaluates the complete loop:

```text
sequence X -> predicted Y_RW -> MSRB -> greedy allocation -> measured Y_RW
```

It can establish four bounded results:

1. frozen component-wise prediction error for $Y_{\mathrm{RW}}$;
2. frozen within-view rank preservation;
3. within-batch view specificity; and
4. descriptive performance relative to the pre-round observed corpus.

For one view, comparing its six nominees with the other twelve candidates
tests within-batch specificity only. The other twelve are not random controls;
they were selected by two correlated MSRB views. This comparison does not
estimate enrichment over the full candidate universe, improvement over random
selection, superiority to another acquisition policy, or a causal benefit
from retraining.

Judging acquisition efficacy requires a comparison rule fixed before outcomes
are observed. The round-0 receipt uses all 296,010 unordered groups of six from
the 27 prior observed labels. For each target view, that exhaustive historical
distribution provides a model-free reference for the best and median observed
MSRB among six candidates. No random seed is needed because every possible
six-candidate group is included.

This reference is deliberately limited. The prior 27 labels are an existing
measured corpus, not a randomized or physically measured control cohort from
the current candidate universe. The comparison can show where the newly
measured six fall relative to historical six-candidate groups. It cannot by
itself establish prospective enrichment, acquisition efficacy, or hill
climbing. The exact inputs, rule, endpoints, artifacts, and claim limits are
frozen in the study-owned `evaluation_baseline.yaml` receipt before outcome
review. The receipt also binds the campaign configuration and allocator
version. Its verifier recomputes all three predicted MSRB score surfaces and
requires the sequence-unique allocation to reproduce the selected 18 exactly.
Tie handling, even-sample medians, undefined rank correlations, and missing
values have fixed rules rather than being chosen after measurement.

Multiple prospectively frozen rounds are needed to test whether retraining
improves the $X\rightarrow Y$ map and policy outcomes. The planned decision
horizon is two to three design-build-test-learn rounds with 18
sequence-unique candidates per round. A no-go decision should use the declared
prospective evidence and baseline, not a flat cumulative maximum or a
six-versus-twelve contrast. This decision belongs to the campaign protocol,
not to MSRB.

### Claim boundaries

For this study, MSRB does not establish:

- feasibility or conformance to a biological specification;
- absolute OFF, reporter background, or absence of expression;
- response latency, transient peaks, adaptation, or other kinetics removed by
  the 4–8-hour reduction;
- whether YFP/CFP changed through YFP induction or CFP loss;
- whether YFP/OD600 changed partly through growth or OD600 reduction;
- growth, viability, metabolic burden, or toxicity;
- cell-population heterogeneity hidden by bulk well measurements;
- a population-average label across experiments or future-experiment
  uncertainty;
- comparable scores across target views or protocol versions;
- calibrated predictive uncertainty;
- graded or don't-care targets, causal stress effects, or biochemical
  interaction;
- diversity beyond exact sequence identity or out-of-distribution safety;
- reliable prospective hill-climb efficacy; or
- synthesis authorization.

pDual-10 is a condition-matched signal reference. The supported OFF claim is
**suppression relative to same-state pDual-10**, not absolute non-expression.

### Authority and sources

| Stage | Authority |
| --- | --- |
| Workbook ingest, event alignment, well reduction, within-experiment bootstrap, and censor provenance | Reader response-window contract |
| Reader alias, candidate, sequence, and BaseRender identity | Study promoter-candidate bindings |
| Repeated-experiment disposition and soft-min evidence cohort | Stress study |
| Candidate-level observed-Y publication | Study response-window label promotion |
| Generic MSRB mathematics and diagnostics | OPAL objective plugin |
| Model fit, prediction, selector, allocation, and campaign ledger | Active OPAL campaign |
| Cross-repository routing | Reader–study–OPAL bridge; routing only |

The bridge routes artifacts. It does not define formulas, aliases, repeat
rules, or campaign policy. OPAL does not import Reader, and Reader does not
resolve candidate authority.

Source map:

- Reader response-window contract:
  `reader/docs/lib/plate_reader/response_window.md`
- Study observation and repeat policy:
  `src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/README.md`
- Study label promotion:
  `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/README.md`
- Study candidate bindings:
  `src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/README.md`
- Active study protocol:
  `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/protocol.yaml`
- Generic objective definition:
  `src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md`
- Objective implementation:
  `src/dnadesign/opal/src/objectives/multistate_response_behavior_math.py`
- Active campaign:
  `src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml`
- Pressure tests and RMF comparison:
  `src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/multistate_response_behavior_shadow/latest/`

The digest-bound workbench bundle is evidence, not an executable campaign. Its
verifier checks the artifact inventory, schemas, derivations, and provenance.
The activation receipt binds the protocol to the reviewed bundle. It permits a
prospective MSRB learning probe but does not authorize synthesis. That separate
authority is recorded for the exact assay-batch-1 package in
`docs/studies/stress_ethanol_cipro_growth/record/synthesis_handoffs.yaml`.
