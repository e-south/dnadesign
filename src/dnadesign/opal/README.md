![OPAL banner](assets/opal-banner.svg)

# OPAL

OPAL (Optimization with Active Learning) uses measured outcomes and sequence
features to choose which designs to test next. It fits a predictive model, ranks
candidates against your objective and selects a batch. Add new measurements to
repeat the cycle. The [documentation](docs/index.md) explains setup and outputs.

## Documentation

### Try a selection round

OPAL accepts an ordinary Parquet table of candidate sequences and numerical
features, plus batches of measured outcomes. SeqLedger is an optional data integration.
From the DNADesign checkout, install only the OPAL runtime and try a round:

```bash
uv sync --locked --no-default-groups --extra opal
source .venv/bin/activate
opal demo-matrix --rounds 0 --keep
```

The activated `opal` command also works outside the checkout. See
[installation choices](../../../docs/setup/installation.md) for the scoring API,
notebook dependencies and other tools.

The included synthetic examples fit predictive models, select five candidates
each and verify the saved selections. They run in fresh temporary directories
and print where to inspect the results. Success ends with `overall_ok: True`.
The [walkthrough](docs/index.md#try-a-complete-selection-round) shows how to
inspect a selected batch and start a campaign with your own data.

### Guides and reference

- [Campaign round](docs/workflows/campaign-round.md): initialize, ingest labels, fit, select and verify.
- [Configuration](docs/reference/configuration.md): candidate data, features, objectives and selection settings.
- [Campaign routes](campaigns/README.md): maintained examples and external campaign placement.
- [SeqLedger feature workflow](docs/workflows/seqledger-infer-x-active-learning.md): use features already written by Infer.
- [Architecture](docs/concepts/architecture.md): campaigns, views, rounds and selection batches.
- [CLI reference](docs/reference/cli.md): commands and flags.
- [Repository docs](../../../docs/README.md): cross-tool workflows.
