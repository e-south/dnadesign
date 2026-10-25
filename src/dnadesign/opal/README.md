![OPAL banner](assets/opal-banner.svg)

OPAL (Optimization with Active Learning) uses measured outcomes and sequence
features to choose which designs to test next. It fits a predictive model, ranks
candidates against your objective and selects a batch. Add new measurements to
repeat the cycle. The [documentation](docs/index.md) explains setup and outputs.

## Documentation

- **Try a selection round:** from a configured checkout, run
  `uv run opal demo-matrix --rounds 0 --keep`. Included synthetic examples select
  five candidates each in fresh temporary directories. Success ends with
  `overall_ok: True`. [Inspect the outputs and use your own data](docs/index.md#try-a-complete-selection-round).
- [Campaign round](docs/workflows/campaign-round.md): initialize, ingest labels, fit, select and verify.
- [Configuration](docs/reference/configuration.md): candidate data, features, objectives and selection settings.
- [Campaign routes](campaigns/README.md): maintained examples and external campaign placement.
- [USR feature workflow](docs/workflows/usr-infer-x-active-learning.md): use features already written by Infer.
- [Architecture](docs/concepts/architecture.md): campaigns, views, rounds and selection batches.
- [CLI reference](docs/reference/cli.md): commands and flags.
- [Repository docs](../../../docs/README.md): cross-tool workflows.
