<!--
---
id: opal-objective-plugins
title: OPAL objective plugins
owner: dnadesign-maintainers
status: active
last_verified: 2026-10-28
---

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[OPAL docs](../../index.md) / [Prediction and selection](../README.md)

# Choose an objective

An objective turns predicted outcomes (Y) into scores that express your
selection goal. Choose it from the meaning, units and order of your outcome
columns. The linked pages define its equations and configuration. A
[selection rule](../selection/README.md) then uses those scores to choose candidates.

The multistate objectives answer different questions using the same ordered
response and reference-relative signal values. RMF measures clearance from
explicit requirements. MSRB ranks ON/OFF behavior while allowing bounded
compensation between requirements. SFXI uses a different logic/intensity input
and rewards setpoint fidelity multiplied by intensity.

## Built-in objective plugins

| Objective | Goal and required outcomes | Score and uncertainty |
| --- | --- | --- |
| [Multistate Response Behavior (MSRB)](multistate-response-behavior.md) | Binary ON/OFF behavior from ordered response and reference-relative signal values; allows bounded compensation | `behavior_score`; no uncertainty |
| [Response-magnitude feasibility (RMF)](response-magnitude-feasibility.md) | Worst clearance from explicit response, ON and OFF requirements | `feasibility_margin`; no uncertainty |
| [Setpoint fidelity × intensity (SFXI)](sfxi.md) | Setpoint match and intensity from an SFXI logic/intensity eight-vector | `sfxi`; optional `sfxi` standard deviation |
| `vector_target_similarity_v1` | Closeness to a target vector in the same units and coordinate order | `negative_mse`; no uncertainty |
| `vector_channel_v1` | One selected outcome from a vector | Configured channel; no uncertainty |
| `scalar_identity_v1` | One scalar outcome used directly as the score | `scalar`; optional `scalar` standard deviation |

The objective's parameters express your priorities: target vectors, thresholds,
scales and setpoints can change candidate rankings. For MSRB, read
[the soft minimum and
scale](multistate-response-behavior.md#why-msrb-uses-a-soft-minimum), [the three
families](multistate-response-behavior.md#three-behavior-families),
[compensation](multistate-response-behavior.md#the-central-compensation-example),
and [claim limits](multistate-response-behavior.md#what-the-score-does-not-establish).

## Channel reference format

Each objective plugin declares local channel names. A selection view references
those local names:

- `selection.params.score_ref = "<score_channel_name>"`
- `selection.params.uncertainty_ref = "<uncertainty_channel_name>"` (required for `expected_improvement`)

`score_ref` resolves only score channels. `uncertainty_ref` resolves only uncertainty channels.
OPAL namespaces persisted channels by selection view ID, so repeated instances
of the same objective plugin cannot collide.

## Common selection wiring examples

Top-N:

```yaml
selection_views:
  - id: primary
    objective: {name: scalar_identity_v1, params: {}}
    selection:
      name: top_n
      params:
        top_k: 12
        score_ref: scalar
        objective_mode: maximize
        tie_handling: competition_rank
```

Pool-relative weighted acquisition (`expected_improvement` registry ID):

```yaml
selection_views:
  - id: primary
    objective: {name: scalar_identity_v1, params: {}}
    selection:
      name: expected_improvement
      params:
        top_k: 12
        score_ref: scalar
        uncertainty_ref: scalar
        objective_mode: maximize
        tie_handling: competition_rank
        alpha: 1.0
        beta: 1.0
```

This selector is available only when the objective publishes a standard
deviation channel. Objective-specific pages document their channel names.
The [GP acquisition demo](../../../campaigns/demo_gp_ei/README.md) runs this
configuration with a predictor that supplies standard deviations.

## Add an objective

Add a plugin when scoring requires different input semantics, equations or
selection preferences. If a scalar outcome already expresses your goal, use
`scalar_identity_v1` and describe the measurement in your data contract and
selection-view name.
