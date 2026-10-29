<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[OPAL](../README.md)

# OPAL documentation

OPAL combines numerical descriptions of candidate sequences with measurements
from tested candidates to choose the next batch to test. Start with the demo to
see the complete loop, then use the campaign guide with your own data.

## Try a complete selection round

From the DNADesign checkout, install the OPAL runtime and activate it:

```bash
# Install the focused runtime and activate its commands.
uv sync --locked --no-default-groups --extra opal
source .venv/bin/activate
# Run the examples and retain their outputs for inspection.
opal demo-matrix --rounds 0 --keep
```

The demo uses synthetic data in fresh temporary directories. It runs
random-forest and Gaussian-process examples, selects five candidates in each,
and checks the selections against the saved predictions. Success ends with
`overall_ok: True` and zero mismatches for each example. The printed `tmp_root`
contains the retained campaigns and their `outputs/` directories.

To inspect the random-forest selection, replace `/path/to/tmp_root` below with
that printed directory:

```bash
# Inspect the candidates selected by the random-forest example.
opal selection-set show \
  -c /path/to/tmp_root/demo_rf_topn/configs/campaign.yaml \
  --view primary --round latest
```

The activated `opal` command also works outside the checkout. This installation
supports local Parquet inputs; SeqLedger is optional. Add `--extra opal-notebook`
when installing the interactive notebook dependencies. See
[installation choices](../../../../docs/setup/installation.md) for details.

## Use your own data

1. [Prepare a candidate table](./reference/data-contracts.md#prepare-a-candidate-table)
   containing sequence IDs, DNA and numeric features.
2. [Run a campaign round](./workflows/campaign-round.md) to import measurements,
   fit a predictor, select candidates and export their DNA. The guide includes
   the next round and recovery after an interrupted run.

If your features are already stored in SeqLedger, use the
[SeqLedger integration](./workflows/seqledger-infer-x-active-learning.md) for input
setup, then follow the same selection loop.

The [example campaigns](../campaigns/README.md) provide complete configurations.
A campaign keeps the candidate pool, measurements and learning history together.
Selection views let you apply different objectives to the same predictions.
See [campaigns, rounds and views](./reference/configuration.md#campaigns-rounds-and-views) before
configuring multiple targets.

## Choose prediction and selection methods

- [Models](./plugins/models/README.md): available predictors and their settings.
- [Gaussian processes](./plugins/models/gaussian-process.md): prediction and uncertainty assumptions.
- [Objectives](./plugins/objectives/README.md): convert predicted outcomes into scores for your goal.
- [Selection rules](./plugins/selection/README.md): rank candidates using scores and, when required, uncertainty.
- [Pool-relative weighted acquisition](./plugins/selection/expected-improvement.md): the rule named `expected_improvement` and its mathematical definition.
- [Feature transforms](./plugins/transforms/x.md) and [measurement transforms](./plugins/transforms/y.md): prepare inputs and training targets.

## Look up a setting or output

- [Configuration](./reference/configuration.md): fields in `campaign.yaml`.
- [CLI commands](./reference/cli.md): inspection, recovery and export options.
- [Data and saved results](./reference/data-contracts.md): candidate types, labels and prediction records.
- [Plots](./reference/plots.md) and [notebooks](./reference/notebooks.md): inspect campaign results.
- [Review manifests](./reference/review-manifests.md): identify and verify a saved review bundle.

## Maintain or extend OPAL

- [Architecture and data flow](./concepts/architecture.md): how configuration reaches fitting, scoring and selection.
- [Round context](./concepts/roundctx.md): runtime state and contract checks.
- [Testing guide](./maintainers/testing-matrix.md): demo coverage, focused tests and repository checks.
