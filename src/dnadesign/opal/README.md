![OPAL banner](assets/opal-banner.svg)

OPAL runs active-learning campaigns over labeled sequence datasets with
explicit feature, objective, selection, and ledger contracts. OPAL v3 fits one
shared phenotype model per campaign, applies named target views to the shared
predictions, and records one final deduplicated selection batch. Start at
the [OPAL docs index](docs/index.md).
**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-13

## Documentation

- [OPAL docs index](docs/index.md): workflows, plugins, concepts, references,
  and maintainer notes.
- [Architecture](docs/concepts/architecture.md): campaign, view, round, and
  selection-batch ownership.
- [Configuration](docs/reference/configuration.md): strict v3 schema.
- [Campaign routes](campaigns/README.md): maintained demos, study-owned
  campaigns, and placement rules.
- [USR infer-derived X workflow](docs/workflows/usr-infer-x-active-learning.md):
  active learning when Infer has already written the feature column.
- [CLI reference](docs/reference/cli.md): command contracts and flags.
- [Repository docs index](../../../docs/README.md): cross-tool workflow routing.
