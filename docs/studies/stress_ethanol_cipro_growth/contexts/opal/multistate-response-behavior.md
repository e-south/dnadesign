---
id: stress-ethanol-cipro-growth-multistate-response-behavior
title: MSRB application for the stress-promoter study
short_name: MSRB
objective_id: multistate_response_behavior_v1
y_space: reader_response_window_vector_v1
owner: stress_ethanol_cipro_growth
status: active
last_verified: 2026-07-18
first_hop: ../../../../../src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md
audience:
  - scientist
  - maintainer
  - operator
  - agent
---

## MSRB application for the stress-promoter study

**Multistate Response Behavior (MSRB)** ranks promoters intended to
implement condition-dependent expression programs by three biological
preferences at once:

1. intended-ON states should have greater reporter response than intended-OFF
   states;
2. intended-ON states should have greater same-state reference-relative signal;
   and
3. intended-OFF states should have lower same-state reference-relative signal.

For this assay, the signal coordinate is reduced YFP/OD600 relative to
same-state pDual-10. That reference is a study binding, not part of the generic
MSRB definition.

The OPAL objective identifier is `multistate_response_behavior_v1`, its scalar
is written $S_{\mathrm{MSRB}}$, and its selectable score channel is
`behavior_score`.

MSRB is the sole ranking objective for the active promoter campaign.
The campaign is a prospectively frozen greedy learning probe. It tests whether
the sequence-to-phenotype model and measured MSRB enrichment improve as new
batches accumulate. Existing retrospective evidence supports the objective's
biological semantics and implementation, but it does not establish reliable
predictive ordering or prospective hill-climb efficacy. Selection and physical
synthesis authorization remain separate decisions.

The [generic OPAL objective contract](../../../../../src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md)
defines the reusable equations and API. The study-owned
[`protocol.yaml`](../../../../../src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/protocol.yaml)
pins the assay, normalization, target masks, model target, selection policy,
and claim boundaries for this study.

### Scope of this study application

Generic MSRB accepts a fixed panel of `K >= 2` named states with binary
intended-ON/OFF membership and an ordered `2K` phenotype. The states need not
represent stress, two perturbations, binary factor codes, or a complete
factorial panel.

This study supplies one concrete binding: four measured conditions, a
condition-matched pDual-10 reference, a 4–8-hour response window, three target
masks, and one active campaign. Those assay and campaign choices do not alter
the generic objective.

```text
sequence representation X
  -> predicted response-window phenotype Y_hat_RW
  -> target mask plus MSRB
  -> behavior score and diagnostics
  -> greedy allocation
```

The model predicts the phenotype, not an MSRB scalar. One predicted phenotype
can therefore be evaluated under several target masks without retraining the
model or changing Reader's assay contract.

### Authority by stage

| Stage | Authority |
| --- | --- |
| Workbook ingest, event alignment, well reduction, within-experiment bootstrap, and censor provenance | Reader response-window contract |
| Reader alias, candidate, sequence, and BaseRender identity | Study promoter-candidate bindings |
| Repeated-experiment disposition and normalization cohort | Stress study |
| Candidate-level observed-Y publication | Study response-window label promotion |
| Generic MSRB mathematics and diagnostics | OPAL objective plugin |
| Model fit, prediction, selector, allocation, and campaign ledger | Active OPAL campaign |
| Cross-repository routing | Reader–study–OPAL bridge; routing only |

No layer recomputes another layer's scientific contract. In particular, OPAL
does not import Reader, Reader does not resolve candidate authority, and the
bridge does not invent formulas, aliases, repeat rules, or campaign policy.

### End-to-end evidence path

```text
OD600, YFP, and CFP well trajectories
  -> event-relative 4-8 h reduction for each well
  -> within-experiment replicate medians
  -> same-state pDual-10 fluorescence subtraction
  -> one Reader experiment-level response-window vector
  -> exact Reader alias-to-candidate binding
  -> study adjudication of repeated experiments
  -> one exact candidate-level observed vector
  -> immutable, manifest-pinned OPAL label event
  -> sequence representation X
  -> predicted response-window vector Y_hat_RW
  -> target-view MSRB score
  -> top-six ranking per view
  -> sequence-unique 18-candidate allocation
  -> prospective measurement and round-over-round evaluation
```

#### 1. Reader response-window reduction

Reader starts from annotated OD600, YFP, and CFP trajectories. It resolves the
intervention interval and expresses time relative to the intervention. The
primary reduction is the geometric log mean from 4 to 8 hours after the event.

For each well $w$, Reader first reduces the trajectory in time:

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

Integrating the log ratio is equivalent to taking a geometric time mean on the
ratio scale. Each well is reduced before wells are combined. Pooling raw time
points across wells would give wells with denser sampling more influence and
would erase the replicate structure.

The 4–8-hour window captures an established stress response while reducing the
late-plate OD accumulation, overflow exposure, and plate-age effects seen in
the longer 6–12-hour window. Alternative windows, event bounds, area-under-the-
curve summaries, and delta summaries remain sensitivity evidence. They do not
silently replace the primary reduction.

Reader's canonical details live in
`reader/docs/lib/plate_reader/response_window.md` in the sibling Reader
repository.

#### 2. Within-experiment replicate handling

For assay state $i$, Reader summarizes technical replicate wells with medians:

$$
r_i = \operatorname{median}_{w \in \mathrm{design},i} R_w,
$$

$$
b_i =
\operatorname{median}_{w \in \mathrm{design},i} F_w
{}-
\operatorname{median}_{w \in \mathrm{pDual\text{-}10},i} F_w.
$$

This creates one estimate per experiment, design, and state. The reference is
both experiment-matched and state-matched. A stressed candidate is never
compared with pDual-10 measured in another condition.

Reader preserves three evidence layers alongside the central estimates:

- within each state, bootstrap draws reuse the same sampled design-well
  indices for $r_i$ and the design contribution to $b_i$; independently
  generated state draws are then assembled into one eight-component draw;
- event-time bounds show sensitivity to intervention-time uncertainty; and
- censor provenance distinguishes exact values from instrument overflow or
  policy bounds.

The bootstrap preserves the measured response/fluorescence pairing within a
state. It does not estimate covariance between states, candidates, or
experiments. These evidence layers are not averaged into the central vector.
For pDual-10 compared with itself, the same resampled wells are reused, so all
central and bootstrap $b_i$ values are definitionally zero rather than
carrying artificial reference uncertainty.

#### 3. Study-owned repeat adjudication and label promotion

One Reader experiment is one evidence unit regardless of how many valid wells
it contains. When a candidate appears in more than one experiment, the study
does not pool all wells, infer authority from the latest timestamp, or average
discordant experiments automatically.

The study applies an explicit disposition:

- a singleton experiment contributes its exact vector;
- an accepted repeated candidate contributes one reviewed Reader experiment;
- a discordant or unresolved candidate is excluded pending remeasurement; and
- any candidate with a bounded primary component is excluded from exact-label
  promotion.

Every contributing experiment, exclusion, Reader manifest digest, candidate
binding, and uncertainty source remains recorded. Cross-experiment differences
stay visible as repeat diagnostics even when one reviewed source supplies the
training label.

The candidate binding is exact and study-owned. A Reader design alias must
resolve to one candidate and a matching sequence digest. Missing, duplicated,
fuzzy, prefix-based, or sequence-mismatched joins fail. Reader aliases never
become candidate authority by themselves.

The current immutable label publication contains 27 exact candidate vectors.
OPAL verifies its `opal.observed_label_promotion.v1` receipt and artifact
digests before reading any label. Across future campaign rounds, all label
events remain in the ledger; the configured `latest_only` policy chooses the
latest exact event per candidate for cumulative training. That round-level
policy is distinct from the study's adjudication of repeated Reader
experiments within one published event.

#### 4. The current four-state response-window phenotype

Generic MSRB consumes `2K` ordered coordinates for any fixed `K >= 2`. This
Reader assay has `K=4`, so its phenotype contains eight values.

The Reader handoff is the objective-neutral response-window phenotype

$$
Y_{\mathrm{RW}}=
[r_{00},r_{10},r_{01},r_{11},b_{00},b_{10},b_{01},b_{11}].
$$

Its exact serialized order is:

```text
[r00, r10, r01, r11, b00, b10, b01, b11]
```

| State | Assay condition |
| --- | --- |
| `00` | No stress |
| `10` | Ethanol |
| `01` | Ciprofloxacin |
| `11` | Ethanol plus ciprofloxacin |

| Coordinate | Meaning |
| --- | --- |
| $r_i$ | Median well-level 4–8-hour reduced $\log_2(\mathrm{YFP}/\mathrm{CFP})$ response in state $i$ |
| $b_i$ | Median design $\log_2(\mathrm{YFP}/\mathrm{OD600})$ minus median same-state pDual-10 value |
| $b_i=0$ | Equal reduced fluorescence to pDual-10 in the same state |
| $b_i=1$ | Twofold higher reduced fluorescence than same-state pDual-10 |
| $b_i=-1$ | Twofold lower reduced fluorescence than same-state pDual-10 |

The vector does not encode a target mask, objective score, uncertainty value,
candidate rank, or synthesis decision. The same $Y_{\mathrm{RW}}$ can be
evaluated by any compatible objective without changing Reader's assay evidence.

#### 5. Sequence-to-phenotype prediction

The active campaign trains one multi-output random forest:

$$
X_{\mathrm{sequence}}
\longrightarrow
\widehat{Y}_{\mathrm{RW}}.
$$

$X_{\mathrm{sequence}}$ is the declared 8,192-component sequence
representation. The training target is the complete observed response-window
vector. The model does not fit one scalar per selection view and does not
predict MSRB directly.

OPAL applies each target mask to the same predicted vector after model fitting:

$$
X_{\mathrm{sequence}}
\longrightarrow \widehat{Y}_{\mathrm{RW}}
\longrightarrow S_{\mathrm{MSRB},v}
\longrightarrow \operatorname{topN}_v.
$$

Changing a target mask changes the interpretation and ranking of predictions;
it does not retrain the raw-Y model. This separation makes prediction errors
observable component by component and avoids creating three disconnected
scalar-model lineages.

The current 100-tree random forest is a frozen prospective baseline, not a
claim that random forests are optimal. Retrospective experiment-held-out rank
support is weak. The first batches test whether the loop learns; they do not
begin from an assumption that it already does.

#### 6. MSRB scoring

The [generic objective contract](../../../../../src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md)
defines the three behavior families and their family-balanced smooth
bottleneck. The stress-study binding supplies only the ordered states, target
masks, and assay-resolution scales.

The two scales answer a practical question: how large is one change that this
assay can resolve reliably in each coordinate type? They put response and
signal on common units before the three behavior families are combined. Reader
bootstrap evidence gives:

| Scale | Rounded value | Meaning |
| --- | ---: | --- |
| $s_R$ | `0.308` log2 | q90 within-experiment uncertainty of declared ON-versus-OFF response contrasts |
| $s_B$ | `0.313` log2 | q90 within-experiment uncertainty of same-state pDual-10-relative signal |

In plain language, a response contrast of about `0.31` log2 or a
reference-relative signal change of about `0.31` log2 counts as one
assay-resolution unit. For example:

- an ON-minus-OFF response difference of `0.616` log2 is about
  `0.616 / 0.308 = +2.0` response-resolution units; and
- an intended-OFF value of `b = -0.626` log2 is about
  `-(-0.626) / 0.313 = +2.0` OFF-suppression units.

This conversion lets the response and signal families share one score even
when their measurement precision differs. On the underlying ratio scale,
`0.31` log2 is about a 1.24-fold change. It remains a resolution convention,
not a desired biological effect size.

These numbers are not biological thresholds, metric weights, or fitted model
parameters. Reader generated 500 joint bootstrap draws for each of 41 exact
candidate-experiment units. The study calculated the uncertainty of the
eligible response contrasts and signal values, then used the 90th percentile
(`q90`) as each scale. Thus 90% of the eligible uncertainty estimates are no
larger than the declared scale. Predictions and selected candidates did not
participate in that calculation.

The active
[`protocol.yaml`](../../../../../src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/multistate_response_behavior/protocol.yaml)
stores the exact machine-readable values and derivation recipe. Values such as
those in the scale table carry additional digits there only so independent
reruns use the identical conversion. Those digits support computational
reproducibility; they do not imply matching biological precision. Rounded
values belong in explanatory prose, while full precision belongs in the
machine-readable contract.

The three active interpretations are:

| View | Mask `[00, 10, 01, 11]` | Intended-ON states |
| --- | --- | --- |
| Ethanol | `[0, 1, 0, 1]` | Ethanol; both stresses |
| Ciprofloxacin | `[0, 0, 1, 1]` | Ciprofloxacin; both stresses |
| AND | `[0, 0, 0, 1]` | Both stresses only |

Scores are comparable only within one view under the same ordered states,
mask, normalization, and objective version. An ethanol score and an AND score
are not measurements on one interchangeable scale.

#### How to read the family landscape

The campaign notebook makes the three scored families visible instead of
asking the scalar to explain itself:

- farther right means stronger intended-ON versus intended-OFF response
  ordering, $S_R$;
- farther up means stronger intended-ON signal relative to same-state
  pDual-10, $S_{\mathrm{ON}}$; and
- redder means stronger intended-OFF suppression relative to same-state
  pDual-10, $S_{\mathrm{OFF}}$.

Under one fixed view, a candidate that moves right, up, and red must receive a
higher $S_{\mathrm{MSRB}}$. Up and right alone are not sufficient because the
color is the third family. A bright, responsive point can still rank poorly
when its intended-OFF states remain bright.

Selected predictions are not expected to be the visually highest measured
points for three separate reasons. First, SpyP, sulAp, pDual-10, and the other
observed labels are assay evidence, not unmeasured DenseGen candidates eligible
for selection. Second, the selector ranks the three-dimensional family
combination rather than either plotted axis alone. Third, the batch allocator
requires sequence uniqueness across views; if an earlier view already claims
a sequence, a later view advances to its next-best unallocated prediction.
The batch table records that replacement explicitly.

The selected-candidate decomposition is the authoritative explanation for one
rank. It shows every state-level coordinate, the three family scores, the hard
bottleneck, and the final smooth score. Zero remains a reference direction,
not a biological acceptance threshold.

The OFF-suppression color scale uses one symmetric linear extent for all three
views: the absolute 99th percentile of the round-0 campaign prediction pool,
`6.866` normalized units. Values beyond that display extent keep
their exact score and remain plotted, but their colors saturate at the fixed
colorbar endpoints. The rectangular colorbar avoids treating saturation as a
new score category; the notebook caption reports how many visible values are
saturated. This view-independent display rule makes differences near the
selected candidates legible without silently removing outliers or changing
the objective.

#### 7. Greedy allocation and prospective measurement

For each view, the `top_n` selector ranks candidates by descending
`behavior_score` and nominates six. MSRB is the objective; `top_n` is the
selector. The policy is deterministic greedy exploitation, not Bayesian
optimization and not uncertainty-aware acquisition.

The campaign then allocates one sequence at a time in the declared view order:
ethanol, ciprofloxacin, AND. If another view has already claimed a sequence,
the current view advances to its next-best unallocated sequence. The batch must
contain exactly six candidates per view and 18 unique sequences or the run
fails.

Every measured candidate is assayed in all four states. It therefore provides
new evidence for all three masks, not only the view that nominated it.

### Applied controls and promotion evidence

SpyP ranks near the top of observed ethanol examples, and sulAp ranks near the
top of observed ciprofloxacin examples. They provide face-validity checks, not
score thresholds or DenseGen architecture matches. The study has no equivalent
positive AND control.

The generic objective tests establish directional monotonicity, Pareto
dominance, state-permutation invariance, equal family standing, bounded
compensation, finite extreme arithmetic, and state-panel pressure from `K=2`
through `K=16`. Study promotion additionally evaluated the three active masks,
Reader bootstrap and event-time sensitivity, repeated experiments, censoring,
normalization sensitivity, and sequence-deduplicated allocation. The
digest-bound activation receipt and shadow bundle preserve those results.

### Uncertainty and censoring

Uncertainty is evidence alongside the eight-value phenotype. It is not a ninth
component and is not part of the current selector.

| Evidence | Meaning | Current role |
| --- | --- | --- |
| Joint state bootstrap | Within-experiment replicate uncertainty; paired response and design-fluorescence resampling within each state | Scale derivation and rank-sensitivity review |
| Event-time bounds | Sensitivity to intervention-time uncertainty | Separate envelope review |
| Repeat disagreement | Between-experiment source dependence for one candidate | Study adjudication and exclusion |
| Censor provenance | Exact, lower-bounded, or upper-bounded assay value | Exact-label gate |
| Model uncertainty | Epistemic uncertainty in $X\rightarrow Y$ | Not calibrated or used by the selector |

Reader bootstrap variation must not be relabeled as model uncertainty. Random-
forest tree dispersion is also not a calibrated posterior. Expected
improvement or another uncertainty-aware acquisition policy would require a
separate validated uncertainty contract.

The active campaign consumes point estimates only. Candidates with bounded
primary response-window values do not become apparently exact labels.

### Required observability

The OPAL notebook must keep the scalar subordinate to its evidence. For every
selection view it provides:

1. a family landscape with response ordering on the x-axis, intended-ON signal
   on the y-axis, and intended-OFF suppression in color;
2. independent prediction-pool, allocated, observed-batch, and label toggles;
3. a selected-candidate decomposition containing every state-level clearance,
   the three family scores, hard bottleneck, and $S_{\mathrm{MSRB}}$;
4. score-versus-rank context for the complete prediction pool;
5. the response-window phenotype summary with mathematical channel labels;
6. BaseRender sequence evidence through its public adapter; and
7. run, model, label, normalization, candidate-binding, allocation, and plot
   digests.

After prospective measurements arrive, the same notebook must add predicted-
versus-observed $Y_{\mathrm{RW}}$ and MSRB evidence. Round-over-round objective
distributions and cumulative best traces become informative after at least two
measured campaign batches; empty or single-round panels should not pretend to
show learning.

### Prospective hill-climb evaluation

The first batch is evidence about the complete loop, not merely the scalar:

```text
sequence X -> predicted Y_RW -> MSRB -> greedy allocation -> measured Y_RW
```

For each view and round, record:

- the six candidates nominated by that view versus the other twelve batch
  candidates scored under the same view;
- their frozen predicted and measured eight-component phenotypes;
- predicted and measured family scores, hard bottlenecks, and MSRB scores;
- objective enrichment relative to the pre-round observed corpus;
- candidate overlap among views and any next-best-unallocated replacements;
- component-wise prediction error and within-view rank preservation; and
- cumulative best measured behavior and distribution shifts across rounds.

One batch can test directional enrichment. Multiple prospectively frozen
rounds are required to assess whether retraining improves the $X\rightarrow Y$
mapping and whether MSRB supports useful hill climbing. Objective semantics can
be correct even when the predictor is weak.

The planned decision horizon is two to three design-build-test-learn rounds,
with 18 sequence-unique candidates in the current campaign contract. The
practical no-go outcome is no repeatable enrichment toward ethanol-responsive,
ciprofloxacin-responsive, or combined-stress behavior: the six candidates
nominated by a view do not outperform the other twelve batch candidates under
that same view, cumulative best measured behavior does not improve, and the
three family diagnostics show no coherent directional gain. This is a
prospective campaign decision, not a threshold embedded in the MSRB formula.

The retrospective shadow evaluation used 27 promoted labels from six source
experiments and 154,785 fixed prediction vectors. MSRB and RMF allocations
overlapped on 15 of 18 sequences. The weakest grouped MSRB ordering was
negative, and normalization sensitivity changed some top-six identities. This
supports a cautious prospective test, not a claim of validated optimization.
The digest-bound comparator evidence remains in the metastudy; RMF is not an
alternate executable stress campaign.

### Claim boundaries

MSRB supports the following claims:

- every desired directional improvement contributes to ranking;
- response ordering, intended-ON signal, and intended-OFF suppression have
  equal family standing;
- poor coordinates receive more influence than already favorable coordinates;
- a favorable outlier has bounded influence; and
- the same predicted phenotype can be interpreted under different fixed masks.

MSRB does not establish:

- feasibility or conformance to a biological specification;
- absolute OFF, reporter background, or absence of expression;
- growth, viability, metabolic burden, or toxicity;
- comparable scores across target views or protocol versions;
- calibrated predictive uncertainty;
- reliable prospective hill-climb efficacy; or
- synthesis authorization.

pDual-10 is a condition-matched fluorescence reference. Therefore the exact
OFF claim is **suppression relative to same-state pDual-10**. No transformation
of $b_i$ can recover an unmeasured reporter-negative background.

### Verification and source map

Scientific and implementation sources are intentionally split by authority:

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
- Generic objective mathematics:
  `src/dnadesign/opal/docs/plugins/objectives/multistate-response-behavior.md`
- Objective implementation:
  `src/dnadesign/opal/src/objectives/multistate_response_behavior_math.py`
- Active campaign:
  `src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml`
- Pre-promotion pressure tests and RMF comparison:
  `src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/multistate_response_behavior_shadow/latest/`

These digest-bound metastudy results remain evidence, not an executable
campaign route. The active protocol authorizes a prospective MSRB learning
probe while preserving the stated nonclaims and synthesis prohibition.

The shadow manifest and decision stay inside the generated workbench bundle.
Its verifier checks the complete artifact inventory, bytes, schemas,
derivations, and provenance as one unit. The activation receipt binds that
bundle's path and digests without copying an incomplete subset into source.
