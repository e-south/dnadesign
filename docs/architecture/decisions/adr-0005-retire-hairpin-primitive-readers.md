## ADR 0005: Retire Cruncher hairpin primitive readers

**Status:** accepted
**Date:** 2026-08-22
**Owner:** dnadesign-maintainers

### Context

Cruncher exposed two run-directory readers that converted Snapback and
scar-nick artifacts into `SnapbackCapPrimitive` and
`ScarNickStemBasePrimitive` objects. They were created as a file-shaped bridge
for the former `dnadesign.msd` compiler and the temporary HOP differential
adapter.

The supported predecessor comparison is now a persisted Research Studies
evidence record. The executable adapter is removed, Research Studies has no
Cruncher import, and `dnadesign.msd` is retired by ADR 0004. A repository-wide
search finds no remaining reader caller; only the two Cruncher package
initializers re-export the reader symbols.

### Decision

Delete both primitive readers and their public exports. Do not retain aliases,
forwarding loaders, historical-schema parsers, or HOP adapters in DNA Design.

This decision removes only the cross-package exchange readers. Snapback,
scar-nick, their command surfaces, and the still-unmatched cassette route have
separate capability boundaries and are not removed by this record.

### Consequences

- Cruncher run directories are no longer a domain-object handoff.
- Historical run artifacts remain evidence but have no current parser contract.
- HOP bundles and study-owned records are the supported molecular handoff.
- Any later Cruncher capability removal must preserve cassette behavior and
  resolve producer-specific visual contracts independently.

### Evidence

- [ADR 0004](adr-0004-retire-msd-compiler.md)
- [HOP v0.1.0a6](https://github.com/e-south/hop-design/releases/tag/v0.1.0a6)
- [Research Studies draft PR #13](https://github.com/e-south/research-studies/pull/13)
