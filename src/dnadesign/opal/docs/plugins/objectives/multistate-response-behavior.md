---
id: opal-objective-multistate-response-behavior-v1
title: Multistate Response Behavior objective
short_name: MSRB
objective_id: multistate_response_behavior_v1
owner: dnadesign-maintainers
status: available
last_verified: 2026-07-19
---

## Multistate Response Behavior (MSRB) `multistate_response_behavior_v1`

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-19

Multistate Response Behavior (MSRB) ranks candidate expression programs across
a fixed panel of measured states. A target specification marks each state as
intended ON or intended OFF. The score then rewards three behaviors together:

- stronger response in every intended-ON state than in every intended-OFF
  state;
- higher signal relative to the reference in intended-ON states; and
- lower reference-relative signal in intended-OFF states.

Under a fixed state panel, target mask, and scale protocol, every favorable
change improves the score. The weakest state-level results have the greatest
influence, but improvements elsewhere still count. There is a fixed ceiling on
how much very strong performance in one family can compensate for weakness in
another. No biological pass/fail threshold is built into the objective.

`behavior_score` is a ranking quantity. It does not establish a biological
acceptance boundary, measurement quality, or predictive accuracy.

### At a glance

| Element | Value |
| --- | --- |
| Human short name | MSRB |
| Objective identifier | `multistate_response_behavior_v1` |
| Input | Two values per state: ordered finite `[r(state...), b(state...)]` |
| Selectable score | `behavior_score`, written $S_{\mathrm{MSRB}}$ |
| Direction | Maximize |
| Required context | Ordered state IDs, ON/OFF membership over `K` states, and two positive assay-resolution scales |
| Interpretive diagnostics | Three family scores, hard bottleneck, limiting coordinate, compensation gap and bound, coordinate weights, and reference-direction status |
| Uncertainty | The score uses a point estimate; assay and model uncertainty are reported separately |

### From a multistate phenotype to one score

MSRB begins with two phenotype coordinates for each measured state:

- $r_i$ summarizes the state-specific regulatory response; higher means a
  stronger response.
- $b_i$ summarizes signal relative to a reference measured in that same state;
  positive means above the reference and negative means below it.

These values may be observed assay reductions or model predictions in the same
phenotype space. Wet-lab time series are first reduced into one response and
one reference-relative signal value per state. The measurement protocol
defines the response reduction and replicate handling; it also defines the
reference used to form $b_i$. MSRB begins only after the phenotype has been
formed.

Let `K` mean the number of measured states. Because each state contributes one
response value and one reference-relative signal value, the ordered phenotype
has two times `K` values (`2 × K`, abbreviated `2K`):

$$
Y=[r_1,\ldots,r_K,b_1,\ldots,b_K].
$$

For four states, `2 × 4 = 8`, so this happens to be an eight-value phenotype:

```text
[r_A, r_B, r_C, r_D, b_A, b_B, b_C, b_D]
```

The width comes from the measured state panel. It is not fixed by MSRB. MSRB
accepts any fixed `K >= 2` when every state is assigned to a nonempty
intended-ON or intended-OFF set.

MSRB computes the score as follows:

```text
ordered phenotype with two values per state
  -> assign states to intended ON or OFF
  -> divide each desired difference by a declared resolution scale
  -> summarize response ordering, ON signal, and OFF suppression separately
  -> combine the three family scores into S_MSRB
```

The phenotype, three family scores, and weakest state-level coordinate remain
available beside the scalar.

### When to use this objective

MSRB fits an experimental objective in which:

- every intended ON response should exceed every intended OFF response;
- every intended ON state should have higher reference-relative signal;
- every intended OFF state should have lower reference-relative signal;
- every improvement in any one of those directions should improve the scalar;
- no biological acceptance threshold should be embedded in the score.

When explicit biological acceptance thresholds exist and every threshold must
be met, [Response-Magnitude Feasibility](response-magnitude-feasibility.md) is
the corresponding non-compensatory objective. MSRB instead ranks continuous,
threshold-free directional improvement.

### Input and target specification

For `K >= 2` explicitly named states, the input has exactly two columns per
state, or `2 × K` columns in total:

```text
[r(state_1), ..., r(state_K), b(state_1), ..., b(state_K)]
```

- `r_i` is a finite reduced response value.
- `b_i` is the finite, same-state reference-relative signal coordinate.
- `state_ids` declares the column order with unique, non-empty identifiers.
- `target_mask` has one binary value per state and contains at least one ON and
  one OFF state.

Any fixed state panel is valid when the mask assigns at least one intended-ON
and one intended-OFF member. `K` counts states, not experimental factors, and
the states need not form a factorial panel. Jointly permuting state IDs,
response columns, signal columns, and mask leaves every score unchanged.

The `2K` phenotype begins after assay-window selection, repeat handling,
reference resolution, state identification, and treatment annotation. MSRB v1
uses binary ON/OFF membership over arbitrary fixed `K`; the measured phenotype
coordinates remain continuous.

Partially ON targets, exact setpoints, ordinal preferences, and don't-care
states require a different objective because they change what counts as
improvement. An exact setpoint, for example, must penalize overshoot and
therefore cannot remain monotonic in every raw coordinate.

### What the same-state reference means

MSRB receives a signed same-state reference-relative coordinate $b_i$. When
the coordinate is formed as a difference, let $F^{red}_{design,i}$ and
$F^{red}_{reference,i}$ denote the reduced design and reference measurements
for one state. Then:

$$
b_i = F^{red}_{design,i}-F^{red}_{reference,i}.
$$

`b_i = 0` denotes equality to the reference, positive values denote greater
signal, and negative values denote lower signal. For a log2 coordinate,
`b_i = -3` means eightfold lower signal than the reference in state $i$. This
supports relative suppression only; it cannot establish background or absolute
non-expression.

### Why assay-resolution scales are needed

An ON-minus-OFF response contrast and a same-state reference-relative signal
may have different repeat precision, even when both use log units. A raw change
of `0.3` therefore need not represent the same assay-resolvable change in both
measurements. Two positive scales put them on a common resolution basis:

- $s_R$ is the declared resolution of one ON-minus-OFF response contrast.
- $s_B$ is the declared resolution of one same-state $b_i$ value.

Both must be positive:

$$
s_R>0,\qquad s_B>0.
$$

Dividing by these scales expresses every desired change in assay-resolution
units. The resulting signed value is called a normalized **clearance**. A
clearance of `+1` means one declared resolution unit in a favorable direction;
`-1` means one unit in an unfavorable direction. Here, clearance means signed
displacement from a reference equality, not clearance from a biological
acceptance boundary. The scales balance measurement resolution, not biological
importance.

The measurement protocol derives and fixes both scales before ranking. They are
neither biological thresholds nor preference weights. Because either scale can
change candidate ranks, record both at full precision for reproducibility and
fix them before inspecting rankings; explanatory prose may use rounded values.

For example, if a response difference is `0.62` and the declared response scale
is `0.31`, the normalized response clearance is about `+2`. The division says
that the favorable difference spans about two assay-resolution units. It does
not say that `0.31` is a biological goal or pass boundary.

### The three behavior families

Let the target-ON and target-OFF sets be:

$$
O = \{i : p_i = 1\}, \qquad F = \{i : p_i = 0\}.
$$

| Behavior being sought | Evidence used | Favorable change |
| --- | --- | --- |
| Correct state ordering | Every intended-ON response minus every intended-OFF response | ON value increases or OFF value decreases |
| Higher ON signal | Reference-relative signal in every intended-ON state | ON signal increases |
| OFF signal suppression | Negative reference-relative signal in every intended-OFF state | OFF signal decreases |

#### 1. Response ordering

Basic question: **Does every intended-ON response value exceed every
intended-OFF response value?**

Every intended ON response is compared with every intended OFF response:

$$
x^R_{ij} = \frac{r_i-r_j}{s_R},
\qquad i \in O,\ j \in F.
$$

A positive value means that one ON response exceeds one OFF response; a
negative value means that pair is ordered the wrong way. The family retains all
`|O| x |F|` pairs. Its smooth summary emphasizes weak pairs, but the summary can
still be positive while one pair is reversed. The hard bottleneck and
`all_reference_directions_met` diagnostic expose that case.

Only between-state response differences matter. Adding the same constant to
every $r_i$ leaves the response-ordering family and final score unchanged. The
aligned $b_i$ values carry the reference-relative signal information.

#### 2. ON signal

Basic question: **When the program should be ON, is its signal higher relative
to the same-state reference?**

Every intended ON state contributes:

$$
x^{ON}_i = \frac{b_i}{s_B},
\qquad i \in O.
$$

A positive value means greater signal than the reference; a negative value
means lower signal. Every intended-ON state contributes independently.

#### 3. OFF signal suppression

Basic question: **When the program should be OFF, is its signal lower than the
same-state reference?**

Every intended OFF state contributes:

$$
x^{OFF}_j = \frac{-b_j}{s_B},
\qquad j \in F.
$$

The minus sign turns lower measured signal into a favorable positive clearance.
More negative `b_j` therefore always improves the score. Every intended-OFF
state contributes independently.

An additional scored `b_ON - b_OFF` family would count much of the ON-signal
and OFF-suppression evidence twice. That contrast may be shown as a diagnostic,
but it is not a fourth scored family.

### Family-balanced smooth bottleneck

Each family first reduces its state-level clearances to one family score. For a
family `G` with normalized clearances `x_c`:

$$
S_G = -\log\left(\frac{1}{|G|}\sum_{c \in G}e^{-x_c}\right).
$$

This smooth minimum stays near the weakest clearance while remaining sensitive
to every clearance. If every clearance in a family equals `+1`, its family
score is exactly `+1`. Here and in the final aggregation, `log` is the natural
logarithm paired with `exp`; this is separate from any log2 transform used in
the assay coordinates.

The three resulting family scores are:

- $S_R$: response-ordering behavior;
- $S_{ON}$: intended-ON signal; and
- $S_{OFF}$: intended-OFF suppression.

The final scalar applies the same smooth bottleneck to those three family
scores with equal one-third standing:

$$
S_{\mathrm{MSRB}} = -\log\left[
\frac{1}{3}\left(
e^{-S_R}+e^{-S_{ON}}+e^{-S_{OFF}}
\right)\right].
$$

Higher $S_{\mathrm{MSRB}}$ is better. The score stays close to a weak family
without discarding improvements in the other two. It is not a linear average
of the family scores that allows one arbitrarily strong family to dominate
without bound, and it is not a hard minimum that ignores every nonlimiting
improvement.

### One complete four-state example

This neutral example uses states `A`, `B`, `C`, and `D`. States `B` and `D` are
intended ON; states `A` and `C` are intended OFF. Both resolution scales are
set to `1` only to keep the arithmetic readable.

The input phenotype is:

```text
Y = [0, 2, 1, 3, -1, 2, -2, 1]
    [rA rB rC rD  bA bB  bC bD]
```

The state-level clearances are:

- response ordering: `[rB-rA, rB-rC, rD-rA, rD-rC] = [2, 1, 3, 2]`;
- intended-ON signal: `[bB, bD] = [2, 1]`; and
- intended-OFF suppression: `[-bA, -bC] = [1, 2]`.

Applying the smooth minimum within each family gives:

$$
S_R=1.760,\qquad S_{ON}=1.380,\qquad S_{OFF}=1.380.
$$

Applying the same family-balanced smooth minimum once more gives:

$$
S_{\mathrm{MSRB}}=1.491.
$$

The weakest individual clearance is `+1`. The final score is higher because
the remaining coordinates are stronger, but their influence is bounded. These
numbers are ranking evidence in resolution units; `1.491` is not a pass grade.

One resolution unit also fixes the smoothness convention; the equation has no
separate temperature parameter.

Family means prevent a family from gaining influence merely because it contains
more coordinates. Repeating every coordinate within a family leaves its score
unchanged; selectively duplicating states changes their weight and is not a
valid way to encode preference.

#### State-space cardinality boundary

The objective accepts any fixed `K >= 2`, but scores are comparable only when
the objective version, ordered state space, target mask, and normalization
protocol are identical. Family balancing does not make one weak coordinate
equally influential across different state-space sizes. A coordinate in family
`G` has prior weight `1/(3|G|)`, so its maximum compensation gap is
`log(3|G|)` resolution units. That bound grows as a family gains distinct
coordinates.

Within the response family, each ON/OFF pair has equal prior weight. The ON and
OFF partitions therefore have equal aggregate influence, but each state in the
smaller partition has more individual leverage because it participates in more
pairs.

Scores from different state spaces should not be compared or interpreted as an
all-state conformance guarantee. As the state panel grows, the hard bottleneck,
limiting coordinate, and coordinate weights become increasingly important for
showing how much influence one weak state retains.

### Why every improvement matters

Each coordinate has positive derivative:

$$
\frac{\partial S_{\mathrm{MSRB}}}{\partial x_c}
=
\frac{w_c e^{-x_c}}
     {\sum_d w_d e^{-x_d}}
> 0,
$$

where `w_c = 1/(3|G|)` for a coordinate in family `G`. Therefore:

- increasing any ON response strictly raises the score;
- decreasing any OFF response strictly raises the score;
- increasing the reference-relative signal in any ON state strictly raises the
  score;
- decreasing the reference-relative signal in any OFF state strictly raises
  the score.

The returned `coordinate_weights` are these candidate-specific derivatives in
normalized units. They sum to one for each candidate and are distinct from the
fixed prior weights `w_c`. A poor coordinate receives exponentially more
influence, so the score remains bottleneck-oriented without the plateaus of a
hard minimum.

### Reading the three-family landscape

The standard family landscape places the response-ordering score on the
x-axis, the intended-ON-signal score on the y-axis, and the intended-OFF-
suppression score in color. Under one fixed target view and scale protocol:

- farther right means better response ordering;
- farther up means stronger intended-ON signal;
- higher on the OFF-suppression color scale means stronger intended-OFF
  suppression; and
- a candidate that is strictly better in all three encodings must have a
  strictly higher behavior score.

Top-right is therefore desirable only when the color is also considered. A
point can be farther right and higher yet rank below another point because its
OFF-suppression family is worse. The scalar is computed from the same three
family scores, so there is no hidden fourth preference; the selected-candidate
decomposition exposes the state-level coordinates that produced each family
score.

In the standard campaign review, measured observations may be overlaid for
assay context but are not members of the prediction pool being ranked. If an
allocation rule prevents the same sequence from occupying more than one view,
a view may receive its next-best unallocated candidate. Prediction,
observation, selection, and allocation therefore remain visually distinct.

An optional interactive view places the same three family scores on the x, y,
and z axes. Color and marker shape distinguish predicted, selected, and
observed points; a fourth continuous color scale would repeat the z-axis. The
complete ledger is the numerical record, the 2D landscape is the publication
summary, and the 3D view supports interactive inspection.

The selector maximizes the smooth bottleneck, not Euclidean distance toward a
plot corner. A point may rank higher despite being less rightward or less high
when its OFF-suppression score is materially better. The three family scores,
rather than visual distance in one projection, explain the rank.

### Compensation is limited, not absent

No continuous scalar can be both strictly increasing in every coordinate and
completely non-compensatory. The behavior objective permits bounded
compensation.

For the hard bottleneck `m = min_c x_c` and the prior weight `w_m` of one
limiting coordinate:

$$
m \le S_{\mathrm{MSRB}} \le m - \log(w_m).
$$

Making already strong coordinates arbitrarily better cannot lift the scalar
more than this finite amount above the weakest clearance. Interpret the score
alongside `hard_bottleneck_clearance`, `compensation_gap`,
`maximum_compensation_gap`, the limiting coordinate, and its weights to see the
remaining tradeoff.

### Boundary examples

These examples use unit resolution scales. They demonstrate the equation and
are not biological acceptance criteria.

- If the response family is `+1`, the ON-signal family is `+100`, and the
  OFF-suppression family is `-1`, then $S_{\mathrm{MSRB}}\approx-0.028$.
  Arbitrarily favorable ON signal does not erase weak OFF suppression. A
  balanced candidate with family scores `[+1, +1, +1]` scores exactly `+1` and
  therefore ranks higher than `[+1, +100, -1]`.
- Starting from all-zero coordinates, making one coordinate with prior weight
  $w$ arbitrarily favorable raises the score only toward $-\log(1-w)$. The
  favorable outlier has finite influence.

### Reference direction is not feasibility

The non-selectable diagnostic `all_reference_directions_met` is true when every
normalized clearance is non-negative:

- every ON response is at least every OFF response;
- every ON signal value is at least its same-state reference;
- every OFF signal value is at most its same-state reference.

This is a reference-direction diagnostic, not a biological acceptance test. A
positive `behavior_score` can occur while one coordinate remains negative.
Neither quantity proves absolute non-expression, assay adequacy, or predictive
accuracy.

### Reported quantities

The only selectable score channel is `behavior_score`, the family-balanced
selection scalar. It is maximized.

The following candidate-aligned diagnostics explain the score but are not
separate selection targets:

- `hard_bottleneck_clearance`: worst individual normalized clearance;
- `compensation_gap`: score minus the hard bottleneck;
- `maximum_compensation_gap`: analytic ceiling from the limiting coordinate's
  prior weight;
- `response_family_score`: smooth response-ordering bottleneck;
- `on_signal_family_score`: smooth ON-signal bottleneck;
- `off_signal_suppression_family_score`: smooth OFF-signal-suppression bottleneck.

The detailed mathematical result also returns:

- every state-level clearance and its stable label;
- every coordinate's fixed family-balanced prior weight;
- each coordinate's bottleneck weight;
- the limiting coordinate index and label;
- `all_reference_directions_met`;
- the exact normalization scales.

Prediction ledgers retain candidate-aligned scalar quantities. State-level
coordinates remain available for evidence tables and decomposition plots.

### Implementation reference

These details support reproducible configuration, validation, visualization,
and source navigation without changing the scientific definition above.

#### Configuration

```yaml
transforms_y:
  name: vector_from_table_v1
  params:
    value_columns: [r_state_a, r_state_b, r_state_c, b_state_a, b_state_b, b_state_c]

selection_views:
  - id: profile_1
    objective:
      name: multistate_response_behavior_v1
      params:
        state_ids: [state_a, state_b, state_c]
        target_mask: [0, 1, 0]
        normalization:
          response_scale: <positive study-issued assay-resolution scale>
          signal_scale: <positive study-issued assay-resolution scale>
    selection:
      name: top_n
      params:
        top_k: 6
        score_ref: behavior_score
        objective_mode: maximize
        tie_handling: ordinal
        require_exact_top_k: true
```

Each selection view binds one target mask and one fixed pair of resolution
scales to the objective.

#### Evidence and numerical boundaries

The score consumes a `2K` point estimate and emits no uncertainty channel.
Bootstrap draws, event-time bounds, repeated observations, censoring, and model
refits remain separate evidence; they are not appended to the `2K` phenotype as
another component or collapsed into a single probabilistic standard deviation.

The objective rejects:

- fewer than two states, duplicate or blank state IDs, and state/mask mismatch;
- non-binary, all-ON, or all-OFF masks;
- prediction matrices that are empty, non-finite, or not exactly `2K` columns;
- missing, extra, non-finite, or non-positive normalization scales;
- undeclared plugin parameters, including a temperature or threshold.

Stable log-sum-exp evaluation prevents exponential overflow. Clearances beyond
floating-point range saturate at a finite numerical guard; this affects only
arithmetic extremes far outside assay-resolution inputs.

#### Review views

OPAL review surfaces for MSRB include:

- `multistate_response_behavior_frontier` for the three family scores;
- `multistate_response_behavior_selected_decomposition` for every coordinate,
  family score, hard bottleneck, and selected score;
- `scatter_score_vs_rank` for pool-wide rank context;
- `vector_summary_heatmap` for the objective-neutral predicted phenotype;
- `observed_objective_over_rounds` once multiple measured rounds exist; and
- an interactive three-family inspector for rotating the same $S_R$, $S_{ON}$,
  and $S_{OFF}$ coordinates shown in the 2D landscape.

Annotations are scoped to one campaign, run, round, and target view so that
predicted, selected, and previously observed candidates are not mixed across
analyses.

#### Responsibility boundaries

- The assay producer defines the measurements, reduction window, replicate
  evidence, censoring, and the meaning of $r_i$ and $b_i$.
- The biological study defines state identities, target membership, reference
  choice, resolution scales, and whether an objective is suitable for a
  campaign.
- OPAL evaluates the published equations, predicts the complete phenotype,
  applies the selected target view, and records ranking and allocation.
- Sequence annotation and other external evidence enter review surfaces only
  through explicit, verified sources; they do not alter the score.

#### Source map

- Public math API: `src/dnadesign/opal/api/multistate_response_behavior.py`
- Objective implementation:
  `src/dnadesign/opal/src/objectives/multistate_response_behavior_v1.py`
- Pure math:
  `src/dnadesign/opal/src/objectives/multistate_response_behavior_math.py`
- Parameter schema: `src/dnadesign/opal/src/config/plugin_schemas.py`
- Objective tests:
  `src/dnadesign/opal/tests/objectives/test_objective_multistate_response_behavior_v1.py`
- Study application example:
  `docs/studies/stress_ethanol_cipro_growth/contexts/opal/multistate-response-behavior.md`
