![OPAL banner](assets/opal-banner.svg)

OPAL runs repeatable sequence-candidate campaigns. A campaign declares its
data, features, labels, objective plugin, model, selection policy, and ledger.
Named views can select against the same fitted predictions without creating a
second campaign lifecycle. Start with the [documentation index](docs/index.md).

## Documentation

- [Campaign round](docs/workflows/campaign-round.md): the complete command path.
- [Architecture](docs/concepts/architecture.md): campaign, view, round, and
  selection-batch ownership.
- [Configuration](docs/reference/configuration.md): strict v3 schema.
- [Campaign routes](campaigns/README.md): maintained demos and placement rules
  for external campaigns.
- [USR infer-derived X workflow](docs/workflows/usr-infer-x-active-learning.md):
  active learning when Infer has already written the feature column.
- [CLI reference](docs/reference/cli.md): command contracts and flags.
- [Repository docs index](../../../docs/README.md): cross-tool workflow routing.
