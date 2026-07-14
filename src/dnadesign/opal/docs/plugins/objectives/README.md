---
id: opal-objective-plugins
title: OPAL objective plugins
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-14
---

## OPAL Objective Plugins

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14

Objective plugins convert shared model predictions into named score and
uncertainty channels. Objective-specific pages define the equations.

`sfxi_v1` and `response_magnitude_feasibility_v1` are independent objective
plugins with different Y contracts. `top_n` and `expected_improvement` are
selection plugins. A selection view binds one objective to one selector; an
objective does not choose candidates, and a selector does not infer objective
identity from Y.

Source modules:

- `src/dnadesign/opal/src/objectives/sfxi_v1.py`
- `src/dnadesign/opal/src/objectives/response_magnitude_feasibility_v1.py`
- `src/dnadesign/opal/src/objectives/spop_v1.py`
- `src/dnadesign/opal/src/objectives/scalar_identity_v1.py`
- `src/dnadesign/opal/src/objectives/vector_channel_v1.py`
- `src/dnadesign/opal/src/objectives/vector_target_similarity_v1.py`

### Channel reference format

Each objective plugin declares local channel names. A selection view references
those local names:

- `selection.params.score_ref = "<score_channel_name>"`
- `selection.params.uncertainty_ref = "<uncertainty_channel_name>"` (required for `expected_improvement`)

`score_ref` resolves only score channels. `uncertainty_ref` resolves only uncertainty channels.
OPAL namespaces persisted channels by selection view ID, so repeated instances
of the same objective plugin cannot collide.

### Built-in objective plugins

### `sfxi_v1`

Use for vec8 SFXI objective scoring (logic fidelity x intensity).

- Score channels:
  - `sfxi` (maximize)
  - `logic_fidelity` (maximize)
  - `effect_scaled` (maximize)
- Uncertainty channels:
  - `sfxi`

### `response_magnitude_feasibility_v1`

Use for `K` ordered response states plus `K` aligned reference-relative
magnitude states after a study has calibrated explicit constraint thresholds
and scales.

- Score channels:
  - `feasibility_margin` (maximize)
  - `response_separation` (maximize)
  - `on_magnitude_floor` (maximize)
  - `off_magnitude_ceiling` (minimize)
- Uncertainty channels:
  - none

### `scalar_identity_v1`

Use when the model output is already a single scalar objective.

- Score channels:
  - `scalar` (maximize)
- Uncertainty channels:
  - none

### `spop_v1`

Use when the model output is one Reader SPOP endpoint scalar.

- Score channels:
  - `spop` (maximize)
- Uncertainty channels:
  - none

### `vector_channel_v1`

Use when a vector target should select on one declared channel.

- Score channels:
  - configured as `<channel_name>` (mode from params)
- Uncertainty channels:
  - none

### `vector_target_similarity_v1`

Use when a vector target should select by closeness to a declared target vector.

- Score channels:
  - `negative_mse` (maximize)
- Uncertainty channels:
  - none

### Objective detail pages

- [SFXI behavior and math](sfxi.md)
- [Response-Magnitude Feasibility (RMF) behavior and math](response-magnitude-feasibility.md)
- [SPOP scalar objective](spop.md)

### Common selection wiring examples

Top-N:

```yaml
selection_views:
  - id: primary
    objective: {name: scalar_identity_v1, params: {}}
    selection:
      name: top_n
      params: {top_k: 12, score_ref: scalar, objective_mode: maximize, tie_handling: competition_rank}
```

Expected improvement:

```yaml
selection_views:
  - id: primary
    objective: {name: sfxi_v1, params: {...}}
    selection:
      name: expected_improvement
      params:
        top_k: 12
        score_ref: sfxi
        uncertainty_ref: sfxi
        objective_mode: maximize
        tie_handling: competition_rank
        alpha: 1.0
        beta: 1.0
```
