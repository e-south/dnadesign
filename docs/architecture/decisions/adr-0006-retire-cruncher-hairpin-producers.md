## ADR 0006: Retire superseded Cruncher hairpin producers

**Status:** accepted
**Date:** 2026-08-22
**Owner:** dnadesign-maintainers

### Context

Cruncher's Snapback and scar-nick families originated the foldback, released-state,
terminal-nick, catalog-search, and review behavior used during the Retron hairpin
study. Their packages also owned commands, workspaces, reports, specialized visual
contracts, BaseRender adapters, and a release-enzyme catalog.

HOP v0.1.0a6 now owns the supported generic molecular behavior. Differential
evidence covers accepted and rejected geometry, released-state projection, route
composition, candidate identity and order, diagnostics, molecular states, and
routed views. Research Studies consumes verified HOP products and no longer imports
Cruncher. The former MSD compiler and file-shaped primitive readers are retired by
ADRs 0004 and 0005.

The cassette family is a different, unmatched design route. It shares generic
nickase catalog mechanics with the retired families but does not depend on their
producer models or release-enzyme surface.

### Decision

Delete the Snapback, scar-nick, and release-enzyme producer packages together with
their commands, app orchestration, dedicated workspaces, tests, docs, artifacts,
and producer-specific visual contracts, adapters, and renderers.

Keep the cassette family and the generic nickase catalog behavior it exercises.
Keep Construct, Folding, and BaseRender as generic services. Rename generic
foldback annotation roles so those services do not retain a retired producer's
vocabulary.

Do not retain command aliases, forwarding imports, compatibility readers,
historical-schema parsers, or fallback execution paths.

### Consequences

- HOP is the only supported authority for generic hairpin molecular derivation.
- Research Studies remains authoritative for Retron identities, lineage, and
  experimental interpretation.
- DNA Design retains generic construct composition, structure assessment,
  rendering, cassette planning, and shared nickase catalogs.
- Historical Cruncher artifacts remain evidence, not executable inputs.
- Removing the producer-specific visual surface prevents BaseRender and Folding
  from becoming accidental second authorities for hairpin state.
- A new hairpin method must enter through HOP rather than resurrecting a Cruncher
  workflow family.

### Evidence

- [ADR 0004](adr-0004-retire-msd-compiler.md)
- [ADR 0005](adr-0005-retire-hairpin-primitive-readers.md)
- [HOP v0.1.0a6](https://github.com/e-south/hop-design/releases/tag/v0.1.0a6)
- [Research Studies draft PR #13](https://github.com/e-south/research-studies/pull/13)
