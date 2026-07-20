---
id: opal-objective-plugins
title: OPAL objective plugins
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-20
---

## OPAL Objective Plugins

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-20

Objective plugins convert shared model predictions into named score and
uncertainty channels. Objective-specific pages define the equations.

An objective name is part of the Y contract. Matching array width does not make
two objectives semantically interchangeable.

`sfxi_v1`, `response_magnitude_feasibility_v1`, and
`multistate_response_behavior_v1` are independent objective plugins. The two
multistate objectives accept the same ordered response/reference-signal shape but
answer different questions: RMF measures clearance from explicit feasibility
boundaries; the behavior objective ranks threshold-free desired behavior.
`top_n` and `expected_improvement` are selection plugins. A selection view binds
one objective to one selector; an objective does not choose candidates, and a
selector does not infer objective identity from Y.

### Channel reference format

Each objective plugin declares local channel names. A selection view references
those local names:

- `selection.params.score_ref = "<score_channel_name>"`
- `selection.params.uncertainty_ref = "<uncertainty_channel_name>"` (required for `expected_improvement`)

`score_ref` resolves only score channels. `uncertainty_ref` resolves only uncertainty channels.
OPAL namespaces persisted channels by selection view ID, so repeated instances
of the same objective plugin cannot collide.

### Built-in objective plugins

| Objective | Required Y meaning | Selected score | Uncertainty | Main preference encoded |
| --- | --- | --- | --- | --- |
| [`multistate_response_behavior_v1`](multistate-response-behavior.md) | Ordered response and aligned reference-relative signal states | `behavior_score` | None | Threshold-free binary ON/OFF behavior with bounded compensation |
| [`response_magnitude_feasibility_v1`](response-magnitude-feasibility.md) | Ordered response and aligned reference-relative magnitude states | `feasibility_margin` | None | Worst clearance from explicit response, ON, and OFF requirements |
| [`sfxi_v1`](sfxi.md) | SFXI-specific logic and intensity vec8 | `sfxi` | `sfxi` SD | Setpoint fidelity multiplied by scaled intensity |
| `vector_target_similarity_v1` | Generic vector in one declared coordinate system | `negative_mse` | None | Unweighted distance to an absolute target |
| `vector_channel_v1` | Generic vector | Configured channel | None | One selected channel only |
| `scalar_identity_v1` | One scalar that already is the objective | `scalar` | None | Passthrough |
| [`spop_v1`](spop.md) | Reader SPOP endpoint scalar | `spop` | None | Passthrough with SPOP identity |

No row in this table is preference-free. Target vectors, selected channels,
thresholds, scales, family priors, and setpoints all express different design
intent. A campaign must choose an objective from the meaning of Y and the
scientific preference, not from shape compatibility alone. For MSRB, read
[the soft minimum and
scale](multistate-response-behavior.md#why-msrb-uses-a-soft-minimum), [the three
families](multistate-response-behavior.md#three-behavior-families),
[compensation](multistate-response-behavior.md#the-central-compensation-example),
and [claim limits](multistate-response-behavior.md#what-the-score-does-not-establish).

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

Pool-relative weighted acquisition (`expected_improvement` registry ID):

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
