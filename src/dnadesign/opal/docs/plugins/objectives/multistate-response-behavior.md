---
id: opal-objective-multistate-response-behavior-v1
title: Multistate Response Behavior
short_name: MSRB
objective_id: multistate_response_behavior_v1
owner: dnadesign-maintainers
status: available
last_verified: 2026-08-09
---

# Multistate Response Behavior (MSRB)

Multistate Response Behavior (MSRB) ranks a measured or predicted phenotype—the
ordered values that describe behavior across measured states—against a fixed
binary ON/OFF target. It rewards three properties: intended-ON responses above
intended-OFF responses, strong intended-ON signal relative to a reference, and
suppressed intended-OFF signal relative to that reference.

MSRB is a deterministic ranking objective. It is not a distance metric, a
feasibility test, or evidence that a predictive model is accurate. Higher is
better only when candidates use the same ordered states, target mask, input
units, and soft-min scale.

### At a glance

| Element | Contract |
| --- | --- |
| Objective | `multistate_response_behavior_v1` |
| Input | Ordered finite `[r(state...), b(state...)]` |
| Target | One binary ON/OFF value for each state |
| Selected value | `behavior_score`, written $S_{\mathrm{MSRB}}$ |
| Direction | Maximize |
| Required scale | One positive `softmin_scale`, $\tau$, that controls compensation |
| Main diagnostics | Three family scores, hard bottleneck, limiting coordinate, compensation bound, and reference-direction status |
| Uncertainty | Separate evidence; it is not an input to the score or to `top_n` |

The calculation follows one path:

```text
ordered phenotype Y
  -> binary target mask
  -> three favorable-direction behavior families
  -> equal total weight for each family
  -> soft minimum
  -> behavior_score
  -> ranking
```

The assay or model supplies the phenotype. The study supplies the target mask
and soft-min scale. MSRB computes `behavior_score`; a selector such as `top_n`
ranks that score.

### Phenotype and target

For `K` measured states, the assay supplies two values per state. A state is
one measured condition in the assay panel.

- $r_i$ is a response coordinate. The assay defines what response means.
- $b_i$ is signal relative to a reference measured in the same state.

The ordered phenotype is

$$
Y=[r_1,\ldots,r_K,b_1,\ldots,b_K].
$$

This is a `2K` phenotype: two values for each of `K` states. A four-state
phenotype contains eight values. `K` counts states, not experimental factors,
and the states do not need to form a factorial design.

The target mask assigns every state to one of two sets. Here, $p_i$ is the
binary mask value for state $i$:

$$
O=\{i:p_i=1\},\qquad F=\{i:p_i=0\}.
$$

$O$ contains intended-ON states and $F$ contains intended-OFF states. Both sets
must be nonempty. MSRB v1 does not represent partially ON states, exact
setpoints, ordinal targets, or don't-care states. Those specifications require
an objective that defines their favorable directions explicitly.

OPAL does not infer state identity, response meaning, reference choice, or
target membership. The assay and study define those inputs.

### Reference-relative signal

Let $\widetilde{F}_{\mathrm{design},i}$ and
$\widetilde{F}_{\mathrm{reference},i}$ denote assay-reduced signal for the
design and reference in state $i$. Then

$$
b_i=\widetilde{F}_{\mathrm{design},i}
-\widetilde{F}_{\mathrm{reference},i}.
$$

$b_i=0$ means equal signal to the reference in that state, $b_i>0$ means
greater signal, and $b_i<0$ means lower signal. If the coordinate is
log2-transformed, $b_i=-3$ means eightfold lower signal than the reference in
that state.

This coordinate supports a same-state relative-signal claim. It cannot
establish reporter background, absolute non-expression, or a universal OFF
state.

### Three behavior families

MSRB rewrites every desired change so that larger is favorable. The equations
call these favorable-direction values *clearances*. They are grouped by the
biological question they answer.

#### Response ordering

Every intended-ON response is compared with every intended-OFF response:

$$
d^R_{ij}=r_i-r_j,\qquad i\in O,\ j\in F.
$$

A positive value means that one intended-ON response exceeds one intended-OFF
response. A negative value means that pair is reversed. This is global
ON-versus-OFF ordering.

Adding the same constant to every $r_i$ leaves response ordering unchanged.
Reference-relative signal is carried by the aligned $b_i$ values.

#### Intended-ON signal

Every intended-ON state contributes

$$
d^{ON}_i=b_i,\qquad i\in O.
$$

Higher intended-ON signal always helps. Positive values are above the
same-state reference; negative values are below it.

#### Intended-OFF suppression

Every intended-OFF state contributes

$$
d^{OFF}_j=-b_j,\qquad j\in F.
$$

The minus sign makes lower intended-OFF signal favorable. A more negative
original $b_j$ therefore produces a larger OFF-suppression clearance.

An additional scored ON-minus-OFF signal family would count much of the same
signal evidence twice. It may be useful as a diagnostic, but it is not a
fourth scored family in MSRB v1.

### Family weights balance unequal counts

Target masks can generate different numbers of values in each family. A view
with two intended-ON and two intended-OFF states, for example, has four
response comparisons but only two ON-signal and two OFF-suppression values.
Giving every value equal weight would make response ordering count for half of
the score solely because it contains more comparisons.

MSRB instead gives each family one third of the total starting weight. If
coordinate $c$ belongs to family $G(c)$, its prior weight is

$$
w_c=\frac{1}{3|G(c)|}.
$$

$|G(c)|$ is the number of coordinates in that family.

For the two-ON, two-OFF example, each response comparison begins with weight
`1/12`, while each ON-signal and OFF-suppression value begins with weight
`1/6`.

These weights solve a counting problem. They are not empirical probabilities,
confidence weights, or the final influence of each coordinate. Equal total
family weight is also a declared preference: it does not prove that the three
families are equally important, reliable, or independent.

One useful analogy is an assessment with three sections. Each section counts
for one third even when the sections contain different numbers of questions.
The weights decide how much each section counts; the soft-min scale introduced
next decides how readily strong answers compensate for a weak one.

### Why MSRB uses a soft minimum

MSRB needs one scalar that remains sensitive to every favorable change without
allowing strong values to hide a weak value without limit.

| Combination rule | Useful property | Limitation |
| --- | --- | --- |
| Hard minimum | No compensation | Only the current weakest value affects the score |
| Arithmetic mean | Every improvement matters | Strong values can hide an arbitrarily poor value |
| Soft minimum | Every improvement matters while the score stays closer to weaker behavior | Compensation remains and requires a declared scale |

The selection score is

$$
S_{\mathrm{MSRB}}
=-\tau\log\left(\sum_c w_c e^{-d_c/\tau}\right).
$$

It is called a *soft minimum* because it approaches the smallest clearance as
$\tau$ approaches zero, yet every finite clearance still affects the score
when $\tau$ is positive. It does not standardize candidates or convert one
measurement type into another.

The three family scores apply the same rule within each family:

$$
S_G=-\tau\log\left(
\frac{1}{|G|}\sum_{c\in G}e^{-d_c/\tau}
\right).
$$

They are $S_R$ for response ordering, $S_{ON}$ for intended-ON signal, and
$S_{OFF}$ for intended-OFF suppression. The complete score can also be written

$$
S_{\mathrm{MSRB}}=-\tau\log\left[
\frac{e^{-S_R/\tau}+e^{-S_{ON}/\tau}+e^{-S_{OFF}/\tau}}{3}
\right].
$$

Natural logarithms are used inside the soft minimum. Multiplying by $\tau$
returns the score to the phenotype's input unit; it does not turn a log2 assay
coordinate into a natural-log measurement.

#### What $\tau$ controls

$\tau$ is one positive scale in the same unit as the clearances. It answers:

> Over what distance may stronger behavior compensate for a weak behavior?

A smaller $\tau$ makes the score more like a hard minimum. A larger $\tau$
makes it more mean-like. The scale is therefore part of the ranking rule, not
a biological threshold, limit of detection, learned model parameter, or
candidate-specific uncertainty penalty.

MSRB v1 requires response differences and reference-relative signal values to
share a meaningful unit. One shared $\tau$ treats a one-unit change the same
way in all three families before weights are applied. Separate scales would
add another preference about how much a response change is worth relative to
a signal change.

Removing the field would not remove this choice. Writing $e^{-d}$ would
silently set $\tau=1$ input unit. A hard minimum or arithmetic mean removes the
parameter only by changing the objective's behavior.

A study may derive $\tau$ from assay evidence or declare another prospective
convention. Whatever its source, it must be fixed before allocation and tested
for rank sensitivity because it can change candidate order. Well-resampling
or other assay evidence used to choose the shared ruler does not become a
ninth phenotype value or a candidate-specific term in the score.

#### The central compensation example

Suppose each family has one listed value and $\tau=1$:

- Candidate A has `[0, 0, 0]`; its behavior score is `0`, and all reference
  directions are met at equality.
- Candidate B has `[-0.1, 1, 1]`; its behavior score is about `0.488`, despite
  one wrong-direction family.

MSRB ranks B above A. This is expected behavior. The selector uses
`behavior_score`; the hard bottleneck and direction-met flag explain the
tradeoff but do not constrain it.

The scale can also reverse a ranking. Compare family values `[0, 2, 2]` and
`[0.5, 0.5, 0.5]`. With $\tau=0.3$, they score about `0.33` and `0.50`, so the
balanced candidate ranks first. With $\tau=1$, they score about `0.86` and
`0.50`, so the strong values compensate enough for the first candidate to
rank first.

### Outputs and interpretation

`behavior_score` is the only selectable score channel. Higher is better under
one fixed protocol. The remaining outputs explain that scalar:

| Output | Question answered |
| --- | --- |
| Three family scores | Which broad behavior is strong or weak? |
| `hard_bottleneck_clearance` | What is the weakest state-level value? |
| `limiting_coordinate_label` | Which state or state pair sets that bottleneck? |
| `all_reference_directions_met` | Are all state-level clearances nonnegative? |
| `compensation_gap` | How far does the smooth score sit above the hard minimum? |
| Coordinate weights | Which values have the greatest local influence for this candidate? |

A positive behavior score is not a pass. A high score can coexist with a
negative hard bottleneck because compensation is allowed. Inspect the score,
three family scores, hard bottleneck, limiting coordinate, and direction-met
status together.

#### Reading the family landscape

The standard landscape uses:

- x-axis: response-ordering family score;
- y-axis: intended-ON-signal family score; and
- color or z-axis: intended-OFF-suppression family score.

Farther right, higher, and toward larger labeled values on the OFF-suppression
colorbar or z-axis are all favorable within one view. A candidate that improves
along all three axes must have a higher behavior score. A point that is only
higher and farther right can still rank below another point when its OFF
suppression is worse. The color is a display encoding of the third family; it
does not represent the physical color of a reporter.

The selector ranks the soft-min scalar, not Euclidean distance to a plot
corner. The three axes also need not be statistically independent. They come
from the same phenotype and may share measurement channels.

### Worked four-state example

Let states `A` and `C` be intended OFF and `B` and `D` intended ON. With
$\tau=1$, use

```text
Y = [0, 2, 1, 3, -1, 2, -2, 1]
    [rA rB rC rD  bA bB  bC bD]
```

The state-level clearances are:

- response ordering: `[2, 1, 3, 2]`;
- intended-ON signal: `[2, 1]`; and
- intended-OFF suppression: `[1, 2]`.

The family scores are

$$
S_R=1.760,\qquad S_{ON}=1.380,\qquad S_{OFF}=1.380,
$$

and the final score is

$$
S_{\mathrm{MSRB}}=1.491.
$$

The hard bottleneck is `1`. The remaining stronger values raise the soft
score above that minimum. `1.491` is a ranking value in the phenotype's input
unit, not a pass grade.

### Mathematical properties

For finite values away from floating-point saturation, every favorable
coordinate change strictly improves the score. The derivative is

$$
\frac{\partial S_{\mathrm{MSRB}}}{\partial d_c}
=\frac{w_c e^{-d_c/\tau}}{\sum_e w_e e^{-d_e/\tau}}.
$$

Within one family, coordinates have equal prior weight, so a poorer coordinate
has greater influence. Across families, influence depends on both the value
and its prior weight. The numerically weakest coordinate is not guaranteed to
have the greatest influence when family sizes differ.

Let $m=\min_c d_c$ be the hard bottleneck and $w_m$ the prior weight of one
limiting coordinate. Then

$$
m\le S_{\mathrm{MSRB}}\le m-\tau\log(w_m).
$$

The compensation gap is finite, but it can be material. Since
$w_m=1/(3|G|)$, the largest possible gap grows as a family gains coordinates.
“Bounded” does not mean small and does not guarantee statewise acceptability.
As state count grows, inspect the bottleneck, limiting coordinate, prior
weight, and compensation gap.

### Example assay binding

One four-state dual-reporter promoter assay publishes

```text
[r00, r10, r01, r11, b00, b10, b01, b11]
```

Here, $r_i$ is a reduced log2(YFP/CFP) response and $b_i$ is reduced
log2(YFP/OD600) relative to a condition-matched reference promoter. The same
eight values can be rescored under several binary masks without retraining the
raw-phenotype model.

The assay window, replicate handling, reference identity, target patterns,
soft-min-scale recipe, and campaign evidence belong to the external study
workspace. This page defines only the reusable score contract.

### What the score does not establish

MSRB does not establish:

- biological feasibility or conformance;
- absolute OFF, reporter background, or non-expression;
- assay quality, repeat stability, or predictive accuracy;
- response latency, transient peaks, adaptation, or other behavior removed by
  an upstream time-window reduction;
- assay mechanisms or biological properties not represented in the input
  phenotype;
- between-experiment uncertainty;
- calibrated model uncertainty or out-of-distribution safety;
- graded setpoints, don't-care states, causal effects, or biochemical synergy;
- candidate diversity; or
- synthesis authorization.

These may be necessary evidence or decision dimensions, but they remain
outside this ranking score.

### When to use MSRB

Use MSRB when all of the following are intended:

- the target is a binary partition of a fixed state panel;
- response ordering, ON signal, and OFF suppression all matter;
- every favorable change should affect rank;
- bounded tradeoffs are acceptable; and
- no biological acceptance threshold should be embedded in the selector.

### Lifecycle and OPAL contract

The objective sits between phenotype prediction and selection:

```text
assay observations
  -> candidate-level phenotype labels
  -> model predicts the complete phenotype
  -> target mask and MSRB score each prediction
  -> selector ranks behavior_score
  -> new measurements return through the assay path
```

One configuration is:

```yaml
selection_views:
  - id: profile_1
    objective:
      name: multistate_response_behavior_v1
      params:
        state_ids: [state_a, state_b, state_c]
        target_mask: [0, 1, 0]
        softmin_scale: <positive value in the clearance unit>
    selection:
      name: top_n
      params:
        top_k: 6
        score_ref: behavior_score
        objective_mode: maximize
        tie_handling: ordinal
        require_exact_top_k: true
```

The objective rejects malformed state spaces, nonbinary or degenerate masks,
nonfinite phenotypes, and a missing or nonpositive scale. Stable log-sum-exp
evaluation prevents overflow. At arithmetic extremes, finite saturation may
make an infinitesimal improvement numerically unchanged; it must not decrease
the score.

The score consumes a point phenotype and emits no uncertainty channel.
Bootstrap draws, event-time bounds, repeated observations, censoring, and
model refits remain separate evidence.

### Responsibility and sources

- The assay producer defines the measurements, reduction, replicate evidence,
  censoring, and meanings of $r_i$ and $b_i$.
- The study defines state identities, target membership, reference choice,
  the soft-min scale, label policy, and whether MSRB is appropriate.
- OPAL evaluates the equations, predicts the complete phenotype, ranks one
  score channel, and records selection and allocation.

Implementation sources:

- Public math API: `src/dnadesign/opal/api/multistate_response_behavior.py`
- Pure math: `src/dnadesign/opal/src/objectives/multistate_response_behavior_math.py`
- Objective plugin: `src/dnadesign/opal/src/objectives/multistate_response_behavior_v1.py`
- Parameter schema: `src/dnadesign/opal/src/config/plugin_schemas.py`
- Tests: `src/dnadesign/opal/tests/objectives/test_objective_multistate_response_behavior_v1.py`
