## Response-Magnitude Feasibility `response_magnitude_feasibility_v1`

**Short name:** RMF
**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-13
**Status:** built-in objective for pressure testing; inactive for production
campaigns until the study promotes a label and calibration contract

These equations are the source of truth for the objective mathematics. Reader or
another assay service owns the measurements that enter the objective. A study
owns state meanings, target masks, calibration, and activation.

### Premise

RMF ranks a candidate by its weakest declared requirement: target-ON responses
must separate from target-OFF responses, target-ON output magnitude must clear
a floor, and target-OFF output magnitude must remain below a ceiling.

The three requirements stay visible. They are not multiplied, so one strong
component cannot compensate for one failed component.

### Ordered input contract

For `K >= 2` explicitly named states, the input has exactly `2K` columns:

```text
[r(state_1), ..., r(state_K), m(state_1), ..., m(state_K)]
```

- `r_i` is a finite reduced response value.
- `m_i` is a finite reference-relative output magnitude from the same state
  and reduction.
- `state_ids` declares the column order.
- `target_mask` has one binary value per state and contains at least one ON and
  one OFF state.

The objective does not choose a time window, infer an intervention, aggregate
replicates, resolve a reference, or attach biological treatment names. It also
does not reinterpret canonical SFXI vec8 values. Those are upstream contracts.

For the stress study, Reader instantiates the generic values as:

```text
r_i = window-reduced log2(YFP / CFP) response
m_i = window-reduced log2(YFP / OD600) for the design
      minus the same-state value for pDual-10
```

Thus the generic word `magnitude` means pDual-10-relative fluorescence in that
assay. It does not mean luminance or fluorophore molecular brightness.

### Raw components

For binary target mask `p`, define the target-ON and target-OFF sets:

$$
O = \{i : p_i = 1\}, \qquad F = \{i : p_i = 0\}.
$$

The worst-state response separation is:

$$
d_{\mathrm{response}}
= \min_{i \in O}(r_i) - \max_{j \in F}(r_j).
$$

The observed target-ON magnitude floor and target-OFF magnitude ceiling are:

$$
f_{\mathrm{on}} = \min_{i \in O}(m_i), \qquad
c_{\mathrm{off}} = \max_{j \in F}(m_j).
$$

Interpretation:

- `response_separation > 0`: every target-ON response exceeds every target-OFF
  response;
- `on_magnitude_floor`: the weakest target-ON output relative to the reference;
- `off_magnitude_ceiling`: the strongest target-OFF output relative to the
  reference.

These values remain in upstream measurement units and are always reported.

### Calibrated feasibility

The campaign supplies three thresholds and three strictly positive scales:

$$
q_{\mathrm{response}} =
\frac{d_{\mathrm{response}}-\tau_{\mathrm{response}}}
     {s_{\mathrm{response}}},
$$

$$
q_{\mathrm{on}} =
\frac{f_{\mathrm{on}}-\tau_{\mathrm{on}}}{s_{\mathrm{on}}},
\qquad
q_{\mathrm{off}} =
\frac{\tau_{\mathrm{off}}-c_{\mathrm{off}}}{s_{\mathrm{off}}}.
$$

The selectable scalar is:

$$
S_{\mathrm{RMF}} =
\min(q_{\mathrm{response}}, q_{\mathrm{on}}, q_{\mathrm{off}}).
$$

`S_RMF >= 0` exactly when all three configured requirements pass. These
standardized values are signed decision margins, not classical z-scores.

Thresholds and scales have no defaults. A study must derive them from assay
variation, reference behavior, and its decision requirements. Before that
calibration is promoted, inspect the three-component Pareto surface rather than
using `S_RMF` for production selection.

### How a target mask changes the score

The measured vector stays fixed. Only the ON/OFF partition changes.

For one record:

```text
r = [0.0, 2.0, 1.0, 3.0]
m = [-1.0, 0.5, 0.2, 0.8]
state_ids = [00, 10, 01, 11]
```

| Target | Mask | Response separation | ON magnitude floor | OFF magnitude ceiling |
| --- | --- | ---: | ---: | ---: |
| Input A | `[0,1,0,1]` | 1.0 | 0.5 | 0.2 |
| Input B | `[0,0,1,1]` | -1.0 | 0.2 | 0.5 |
| AND | `[0,0,0,1]` | 1.0 | 0.8 | 0.5 |
| OR | `[0,1,1,1]` | 1.0 | 0.2 | -1.0 |

The same candidate can therefore be useful for one target and fail another.
This is the intended setpoint direction that canonical SFXI did not express
strongly enough in the current stress-study rankings.

### Monotonic top-K selection within one target

For one fixed target mask and calibration, every signed margin has the same
direction: larger values indicate greater clearance from the configured
boundary.

- increasing a target-ON response cannot decrease `q_response`;
- decreasing a target-OFF response cannot decrease `q_response`;
- increasing target-ON fluorescence cannot decrease `q_on`;
- decreasing target-OFF fluorescence cannot decrease `q_off`.

Because `S_RMF` is the minimum of those three margins, improving any required
measurement in its desired direction cannot make the score worse. A `top_n`
selector therefore ranks candidates by the largest weakest requirement. This
is the operative hill-climbing rule:

```text
target mask -> three signed margins -> minimum margin -> descending top-K
```

The rule intentionally has plateaus. Improving a component that is not the
minimum does not change the score until it becomes decision-limiting. Review
plots must therefore show all three components, mark the limiting component,
identify zero as the configured boundary, and display the ON/OFF target
partition.

Changing the target mask changes the ON/OFF partition and therefore changes
the objective. RMF values from different masks are not commensurate, even when
they were calculated from the same measured vector.

### Generality and limits

The equations support any finite ordered state panel with a binary target and
complete aligned response/magnitude values. They do not require two factors or
a complete factorial design. For `K` states, the input has `2K` values.

The objective does not support graded targets, continuous preferences,
all-ON/all-OFF masks, missing states, or direct comparison of scores between
different masks. A separate objective is required for those cases.

Hard minima and maxima implement the literal requirements "every ON" and
"every OFF." They also make one noisy state decisive. This sensitivity grows
with the number of states and with unequal ON/OFF cardinality. Assay bootstrap,
mask-cardinality simulation, and rank-stability review are therefore required
before expanding beyond the four-state stress-study use case.

RMF is monotone in the desired component directions, but it has plateaus:
improving a non-limiting component does not change the minimum. It is a
feasibility hill-climbing score, not a smooth utility function.

### Hill-climbing claim boundary

Positive exemplars are not required to define a signed direction. A negative
score still orders candidates by movement toward the declared feasible region.
That mathematical ordering does not prove that the model predicts the
direction.

Greedy `top_n` is a valid prespecified selection policy when a study accepts
model risk and reports it plainly. Held-out ordering, rank stability, and X
applicability strengthen confidence but do not change RMF mathematics. A
prospective hill climb exists only after selected candidates are measured and
improve the prespecified RMF components relative to the prior corpus.

### Configuration

```yaml
transforms_y:
  name: vector_from_table_v1
  params:
    value_columns: [r00, r10, r01, r11, b00, b10, b01, b11]

selection_views:
  - id: factor_a
    objective:
      name: response_magnitude_feasibility_v1
      params:
        state_ids: ["00", "10", "01", "11"]
        target_mask: [0, 1, 0, 1]
        calibration:
          response_separation_min: <declared value>
          on_magnitude_min: <declared value>
          off_magnitude_max: <declared value>
          response_separation_scale: <positive declared value>
          on_magnitude_scale: <positive declared value>
          off_magnitude_scale: <positive declared value>
    selection:
      name: top_n
      params:
        top_k: 6
        score_ref: feasibility_margin
        objective_mode: maximize
        tie_handling: competition_rank
```

The placeholders are intentional. Missing calibration or state identity is a
configuration error.

### Uncertainty and fail-fast behavior

The plugin emits no uncertainty channel. Expected improvement is not supported.
Assay uncertainty belongs upstream and model stability belongs in grouped
validation or refit analysis.

The objective rejects:

- fewer than two states, duplicate or blank `state_ids`, and state/mask length
  mismatch;
- non-binary, all-ON, or all-OFF masks;
- any prediction matrix that is not finite and exactly `2K` columns wide;
- missing, extra, or non-finite calibration values;
- non-positive calibration scales.

It does not infer missing state identity, thresholds, or assay semantics and
does not fall back to SFXI.

### Source map

- Public math API: `src/dnadesign/opal/api/response_magnitude_feasibility.py`
- Objective implementation:
  `src/dnadesign/opal/src/objectives/response_magnitude_feasibility_v1.py`
- Pure math:
  `src/dnadesign/opal/src/objectives/response_magnitude_feasibility_math.py`
- Parameter schema: `src/dnadesign/opal/src/config/plugin_schemas.py`
- Objective tests:
  `src/dnadesign/opal/tests/objectives/test_objective_response_magnitude_feasibility_v1.py`
- Reader assay contract: `reader/docs/lib/plate_reader/response_window.md`
- Stress-study decision:
  `docs/studies/stress_ethanol_cipro_growth/contexts/opal/response-magnitude-feasibility.md`
