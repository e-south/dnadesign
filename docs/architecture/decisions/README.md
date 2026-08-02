## Architecture Decision Records

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-01

### At a glance
ADRs capture accepted architectural decisions and their consequences.
Use this directory for new decisions going forward.

### Contents
- [ADR naming convention](#adr-naming-convention)
- [ADR lifecycle](#adr-lifecycle)
- [Template](#template)

### ADR naming convention
- Filename format: `adr-XXXX-<short-kebab-title>.md`
- Example: `adr-0001-ci-lane-model.md`
- Numbering applies to new ADRs going forward; historical backfill is optional and can be done later.

### ADR lifecycle
1. Draft from an accepted proposal/execution result.
2. Link proposal, implementation PR(s), and follow-ups.
3. Mark status as `proposed`, `accepted`, or `superseded`.

### Template
- `../../templates/records/adr.md`

### Records
- [ADR 0001: Namespace-scoped compatibility hashes for USR overlays](adr-0001-usr-namespace-contract-hash.md)
- [ADR 0002: Generic linear ssDNA composition in Construct](adr-0002-generic-linear-ssdna-composition.md)
- [ADR 0003: TriJunction as a peer three-way-junction planner](adr-0003-trijunction-peer-tool.md)
