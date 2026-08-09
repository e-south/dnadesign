---
id: opal-campaign-round
title: OPAL campaign round
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
---

# Campaign round

**Type:** workflow
**Owner-boundary:** opal
**Entry artifact:** one valid `opal.campaign.v3` config and its declared records
**Exit artifact:** verified round outputs and ledger rows

OPAL has one execution loop. A campaign chooses plugins for features, targets,
modeling, scoring, and selection, but those choices do not create another
lifecycle.

## Configure the contracts

A campaign config names each plugin and connects the channels explicitly:

```yaml
schema_version: opal.campaign.v3  # Select the strict public config contract.
transforms_x: {name: identity, params: {}}  # Use an already prepared feature vector.
transforms_y: {name: scalar_from_table_v1, params: {y_column: y}}  # Read one declared label column.
model: {name: random_forest, params: {}}  # Fit the named model plugin.
selection_views:  # Bind objective and selector channels explicitly.
  - id: primary  # Name the persisted view.
    objective: {name: scalar_identity_v1, params: {}}  # Preserve a caller-owned scalar score.
    selection:  # Rank this view through one selector.
      name: top_n  # Use deterministic score ordering.
      params:  # Declare every selector input.
        top_k: 8  # Request eight candidates.
        score_ref: scalar  # Read the objective's scalar channel.
        objective_mode: maximize  # Prefer larger values.
        tie_handling: competition_rank  # Keep boundary ties explicit.
```

This scalar example shows the wiring without assigning scientific meaning to
the score. Replace the plugins only when the new contracts agree on target
shape, score channels, uncertainty channels, and direction.

`expected_improvement` requires both `score_ref` and `uncertainty_ref`. The
referenced uncertainty must be a standard deviation produced by the configured
objective. OPAL stops if the channel is absent or incompatible.

## Run one round

From the campaign directory:

```bash
# Check the config and input records without writing campaign state.
uv run opal validate -c configs/campaign.yaml
# Create the campaign state and output directories.
uv run opal init -c configs/campaign.yaml
# Record the labels observed for round zero.
uv run opal ingest-y -c configs/campaign.yaml --round 0 --csv <labels-file> --apply
# Fit, score, and select for the same label cutoff.
uv run opal run -c configs/campaign.yaml --round 0
# Check the saved view against the prediction ledger.
uv run opal verify-outputs -c configs/campaign.yaml --view <selection-view-id> --round latest
```

`validate` is read-only. `init` creates campaign state. `ingest-y` records the
labels visible to a round. `run` fits, scores, and selects. `verify-outputs`
checks the persisted selection against the ledger.

Use `opal guide -c configs/campaign.yaml --format markdown` for commands and
paths resolved from a particular config. Use `opal explain` before a run when
you need a read-only account of its label cutoff and preconditions.

## Keep scientific meaning with its owner

An objective plugin may implement a reusable calculation. Its target mask,
thresholds, calibration, and interpretation still belong to the campaign or
study that chose them. OPAL stores those declarations and applies them; it does
not turn one objective into the meaning of every campaign.

Study workspaces keep their objective policy, evidence, and campaign config
together. They call this same OPAL surface without placing private campaign
state or scientific rationale in the public package.

## Continue

- [Configuration](../reference/configuration.md)
- [Objective plugins](../plugins/objectives/README.md)
- [Model plugins](../plugins/models/README.md)
- [Selection plugins](../plugins/selection/README.md)
- [CLI reference](../reference/cli.md)
