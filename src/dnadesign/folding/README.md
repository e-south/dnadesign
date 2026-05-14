![folding banner](assets/folding-banner.svg)

Folding predicts secondary structure for assembled sequence artifacts and
publishes native ViennaRNA plots for QA. It is a stateless service surface:
producers own the bundle, Folding owns backend preflight, prediction, parsing,
and plot publication.

## Documentation

- [Folding docs](docs/README.md): command routes, artifact flow, backend policy,
  and boundaries.
- [Repository docs index](../../../docs/README.md): cross-tool routing for
  workspace tools, stateless services, and shared contracts.
- [Composition spec](../../../docs/dev/plans/2026-05-13-generic-linear-ssdna-composition-spec.md):
  design rationale for Construct/Folding/BaseRender handoffs.
