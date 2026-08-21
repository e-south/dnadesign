![folding banner](assets/folding-banner.svg)

Folding predicts secondary structure for assembled sequence artifacts and
renders ViennaRNA plots for review. Producers own sequences and bundles;
Folding owns ViennaRNA preflight, prediction, parsing, and plot publication.
The result record is backend-neutral, but the current runner is not a plugin
framework: its only implemented backend is ViennaRNA. Its request contract
supports only `temperature_c`; unknown parameters fail validation instead of
being recorded without effect.

## Documentation

- [Docs index](docs/README.md): command routes, artifact flow, backend policy,
  and boundaries.
- [Composition handoff](../../../docs/architecture/decisions/adr-0002-generic-linear-ssdna-composition.md):
  design rationale for Construct/Folding/BaseRender handoffs.
- [Repository docs index](../../../docs/README.md): cross-tool routing for
  workspace tools, stateless services, and shared contracts.
