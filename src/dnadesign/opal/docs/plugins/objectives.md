## OPAL Objective Plugins

This page documents objective plugin wiring and channel reference rules.
For detailed objective equations and behavior, use objective-specific pages.

Source modules:

- `src/dnadesign/opal/src/objectives/sfxi_v1.py`
- `src/dnadesign/opal/src/objectives/scalar_identity_v1.py`

### Channel reference format

Selection reads channels by explicit reference:

- `selection.params.score_ref = "<objective_name>/<score_channel_name>"`
- `selection.params.uncertainty_ref = "<objective_name>/<uncertainty_channel_name>"` (required for `expected_improvement`)

`score_ref` resolves only score channels. `uncertainty_ref` resolves only uncertainty channels.

### Built-in objective plugins

### `sfxi_v1`

Use for vec8 SFXI objective scoring (logic fidelity x intensity).

- Score channels:
  - `sfxi_v1/sfxi` (maximize)
  - `sfxi_v1/logic_fidelity` (maximize)
  - `sfxi_v1/effect_scaled` (maximize)
- Uncertainty channels:
  - `sfxi_v1/sfxi`

### `scalar_identity_v1`

Use when the model output is already a single scalar objective.

- Score channels:
  - `scalar_identity_v1/scalar` (maximize)
- Uncertainty channels:
  - none

### Draft objective design notes

- [SPOP objective draft](./objective-spop.md)

### Objective detail pages

- [SFXI behavior and math](./objective-sfxi.md)
- [SPOP draft behavior and math](./objective-spop.md)

### Common selection wiring examples

Top-N:

```yaml
selection:
  name: top_n
  params:
    top_k: 12
    score_ref: "scalar_identity_v1/scalar"
    objective_mode: maximize
    tie_handling: competition_rank
```

Expected improvement:

```yaml
selection:
  name: expected_improvement
  params:
    top_k: 12
    score_ref: "sfxi_v1/sfxi"
    uncertainty_ref: "sfxi_v1/sfxi"
    objective_mode: maximize
    tie_handling: competition_rank
    alpha: 1.0
    beta: 1.0
```

### See also

- [SFXI behavior and math](./objective-sfxi.md)
- [Selection plugins](./selection.md)
- [Workflow guides](../index.md#workflows)
