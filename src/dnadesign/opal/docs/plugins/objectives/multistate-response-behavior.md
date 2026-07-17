---
id: opal-objective-multistate-response-behavior-v1
title: Multistate Response Behavior objective
owner: dnadesign-maintainers
status: available
last_verified: 2026-07-17
---

## Multistate Response Behavior `multistate_response_behavior_v1`

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-17

`multistate_response_behavior_v1` is a threshold-free OPAL objective. It ranks
how consistently a candidate moves in the desired directions across a binary
multistate target. It is not a feasibility test, a selector, a label contract,
or a statement that synthesis is authorized.

The plugin contract is available for evaluation. This status does not activate
it in any study campaign; campaign activation remains a study-owned decision.

This page is the source of truth for the objective mathematics and output
semantics. An assay service owns the measurements. A study owns the state
meanings, target mask, assay-resolution scales, and decision to activate the
objective. OPAL owns the equations and exposes their result to a configured
selector.

### When to use this objective

Use this objective when the desired behavior is:

- every intended ON response should exceed every intended OFF response;
- every intended ON state should have greater reference-relative signal;
- every intended OFF state should have lower reference-relative signal;
- every improvement in any one of those directions should improve the scalar;
- no biological acceptance threshold should be embedded in the score.

Use [Response-Magnitude Feasibility](response-magnitude-feasibility.md) when a
study has explicit acceptance thresholds and needs a non-compensatory signed
margin. The two objectives answer different questions. The behavior objective
does not replace or reinterpret RMF records.

### Ordered input contract

For `K >= 2` explicitly named states, the input has exactly `2K` columns:

```text
[r(state_1), ..., r(state_K), b(state_1), ..., b(state_K)]
```

- `r_i` is a finite reduced response value.
- `b_i` is the finite, same-state reference-relative signal coordinate.
- `state_ids` declares the column order with unique, non-empty identifiers.
- `target_mask` has one binary value per state and contains at least one ON and
  one OFF state.

The objective does not require two factors or a complete factorial design. It
uses only the ordered states and their ON/OFF membership. Jointly permuting the
state IDs, response columns, signal columns, and mask leaves every score
unchanged.

The objective does not choose an assay window, combine experimental repeats,
resolve a reference, infer state identity, or attach treatment names. It also
does not consume SFXI vec8 values.

### The three behavior families

Let the target-ON and target-OFF sets be:

$$
O = \{i : p_i = 1\}, \qquad F = \{i : p_i = 0\}.
$$

The study supplies two strictly positive assay-resolution scales:

$$
s_R > 0, \qquad s_S > 0.
$$

They put response and reference-relative signal changes into comparable
resolution units. They are not biological pass/fail thresholds and are not
fitted by the objective.

#### 1. Response ordering

Every intended ON response is compared with every intended OFF response:

$$
x^R_{ij} = \frac{r_i-r_j}{s_R},
\qquad i \in O,\ j \in F.
$$

A positive value means that one ON response exceeds one OFF response. The full
family contains `|O| x |F|` pairwise clearances, so no response state is dropped
by an early minimum or maximum.

#### 2. ON signal

Every intended ON state contributes:

$$
x^{ON}_i = \frac{b_i}{s_S},
\qquad i \in O.
$$

A positive value means that the candidate has greater signal than its declared
same-state reference.

#### 3. OFF signal suppression

Every intended OFF state contributes:

$$
x^{OFF}_j = \frac{-b_j}{s_S},
\qquad j \in F.
$$

A positive value means that the candidate has lower signal than its declared
same-state reference. More negative `b_j` always improves the score.

An additional scored `b_ON - b_OFF` family is intentionally absent. It would
mostly count the ON-signal and OFF-signal-suppression evidence twice. Such a
contrast can be displayed as a diagnostic without changing the selector.

### Family-balanced smooth bottleneck

For a family `G` with normalized clearances `x_c`, define its smooth bottleneck:

$$
S_G = -\log\left(\frac{1}{|G|}\sum_{c \in G}e^{-x_c}\right).
$$

The selection scalar gives the response, ON-signal, and OFF-signal-suppression
families equal prior standing:

$$
S_{behavior} = -\log\left[
\frac{1}{3}\left(
\frac{1}{|R|}\sum_{c \in R}e^{-x_c}
+ \frac{1}{|ON|}\sum_{c \in ON}e^{-x_c}
+ \frac{1}{|OFF|}\sum_{c \in OFF}e^{-x_c}
\right)\right].
$$

The temperature is fixed at one normalized resolution unit. There is no
objective parameter for temperature. A study must freeze the two resolution
scales before comparing candidates; it must not tune the scales or an implicit
temperature to preserve preferred nominations.

Family means matter. Weighting each coordinate equally would give response
pairs or the larger mask partition more influence merely because they contain
more coordinates. Repeating an identical coordinate within one family leaves
that family's mean and the overall score unchanged.

#### State-space cardinality boundary

The objective accepts any fixed `K >= 2`, but scores are comparable only when
the objective version, ordered state space, target mask, and normalization
protocol are identical.
Family balancing does not make one weak coordinate equally influential across
different state-space sizes. A coordinate in family `G` has prior weight
`1/(3|G|)`, so its maximum compensation gap is `log(3|G|)` resolution units.
That bound grows as a family gains distinct coordinates.

Consequently, a study must not compare scores from different state spaces or
interpret this objective as an all-state conformance guarantee. For a larger
fixed state space, promotion evidence must pressure-test weak-coordinate
influence and retain the hard bottleneck, limiting coordinate, and coordinate
weights as mandatory review diagnostics. Those diagnostics expose the tradeoff;
they do not constrain the selector.

### Why every improvement matters

Each coordinate has positive derivative:

$$
\frac{\partial S_{behavior}}{\partial x_c}
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

The returned `coordinate_weights` are these derivatives in normalized units.
They sum to one for each candidate. A poor coordinate receives exponentially
more influence, so the score remains bottleneck-oriented without the plateaus
of a hard minimum.

This strictness is a property of the real-valued equation. In finite-precision
arithmetic, an extremely favorable, nonlimiting coordinate can have an
exponential weight that underflows to zero, so a further change may leave the
returned float unchanged. Assay-scale property tests require strict directional
improvement; arithmetic-extreme tests require finite, nondecreasing behavior
rather than a bytewise strict-increase guarantee.

### Compensation is limited, not absent

No continuous scalar can be both strictly increasing in every coordinate and
completely non-compensatory. The behavior objective permits bounded
compensation.

For the hard bottleneck `m = min_c x_c` and the prior weight `w_m` of one
limiting coordinate:

$$
m \le S_{behavior} \le m - \log(w_m).
$$

Making already strong coordinates arbitrarily better cannot lift the scalar
more than this finite amount above the weakest clearance. The plugin reports
`hard_bottleneck_clearance`, the limiting coordinate index, and its bottleneck
weight so reviewers can see that tradeoff directly.

### Natural zero is not feasibility

The optional diagnostic `all_reference_directions_met` is true when every
normalized clearance is non-negative:

- every ON response is at least every OFF response;
- every ON signal value is at least its same-state reference;
- every OFF signal value is at most its same-state reference.

This is a reference-direction diagnostic, not a biological acceptance test. A
positive `behavior_score` can occur while one coordinate remains negative.
Neither a positive score nor `all_reference_directions_met` means that a study
has established feasibility, non-expression, safety, model support, or
synthesis readiness.

### Same-state reference claim boundary

The generic objective receives `b_i`; it does not know the reference's biological
identity. In the stress promoter study, Reader defines:

$$
b_i = \log_2\left[
\frac{(YFP/OD600)_{design,i}}
     {(YFP/OD600)_{pDual-10,i}}
\right].
$$

In that study, `b_i = -3` means eightfold lower fluorescence than pDual-10 in
the same measured state. The defensible claim is therefore "OFF suppression
relative to same-state pDual-10." pDual-10 does not measure reporter background
or prove absolute non-expression. No objective can recover an unmeasured
background reference.

### Output contract

The only selectable score channel is `behavior_score`, the family-balanced
selection scalar. It is maximized.

The plugin records these candidate-aligned diagnostics, but OPAL does not
advertise them as selectable score channels:

- `hard_bottleneck_clearance`: worst individual normalized clearance;
- `response_family_score`: smooth response-ordering bottleneck;
- `on_signal_family_score`: smooth ON-signal bottleneck;
- `off_signal_suppression_family_score`: smooth OFF-signal-suppression bottleneck.

The public mathematics API additionally returns:

- every state-level clearance and its stable label;
- each coordinate's bottleneck weight;
- the limiting coordinate index and label;
- `all_reference_directions_met`;
- the exact normalization scales.

The plugin persists only candidate-aligned scalar channels and numeric
diagnostics in prediction ledgers. Full coordinate matrices remain available
through the public mathematics API for study-owned evidence tables and plots.

### Configuration

```yaml
transforms_y:
  name: vector_from_table_v1
  params:
    value_columns: [r00, r10, r01, r11, b00, b10, b01, b11]

selection_views:
  - id: factor_a
    objective:
      name: multistate_response_behavior_v1
      params:
        state_ids: ["00", "10", "01", "11"]
        target_mask: [0, 1, 0, 1]
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

This example documents the executable contract. Adding the objective to a
campaign config is a separate study decision. The built-in plugin does not
activate or migrate a campaign.

### Uncertainty and fail-fast behavior

The objective consumes point estimates and emits no uncertainty channel.
Reader bootstrap draws, event-time bounds, repeated experiments, censoring,
and model refits must be evaluated as separate evidence rather than folded into
the eight-value input or treated as a probabilistic standard deviation.

The objective rejects:

- fewer than two states, duplicate or blank state IDs, and state/mask mismatch;
- non-binary, all-ON, or all-OFF masks;
- prediction matrices that are empty, non-finite, or not exactly `2K` columns;
- missing, extra, non-finite, or non-positive normalization scales;
- undeclared plugin parameters, including a temperature or threshold.

Stable log-sum-exp evaluation prevents exponential overflow. Normalized
clearances beyond floating-point range saturate at a finite numerical guard;
this affects only arithmetic extremes far outside assay-resolution inputs.

### Source map

- Public math API: `src/dnadesign/opal/api/multistate_response_behavior.py`
- Objective implementation:
  `src/dnadesign/opal/src/objectives/multistate_response_behavior_v1.py`
- Pure math:
  `src/dnadesign/opal/src/objectives/multistate_response_behavior_math.py`
- Parameter schema: `src/dnadesign/opal/src/config/plugin_schemas.py`
- Objective tests:
  `src/dnadesign/opal/tests/objectives/test_objective_multistate_response_behavior_v1.py`
