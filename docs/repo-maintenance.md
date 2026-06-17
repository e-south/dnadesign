## Repo Maintenance

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-17

Use this page for repo-wide maintenance orientation. Keep durable procedures in
their owner docs and use this page as a compact route map.

### Maintenance Surfaces

| Need | Route |
| --- | --- |
| CI, test scope, coverage, and local parity | [Developer docs](dev/README.md) |
| Cross-tool ownership and information architecture | [Architecture](../ARCHITECTURE.md) |
| Engineering invariants and failure posture | [Design](../DESIGN.md) |
| Operational reliability model | [Reliability](../RELIABILITY.md) |
| Quality gates and entropy reports | [Quality docs](quality/README.md) |
| Current maintainer audit notes | [Monorepo organization audit](dev/audits/monorepo-organization.md) |

### Cadence

- Use targeted validation during ordinary agent work.
- Use full standard tests and coverage gates for broad refactors, dependency
  changes, CI changes, and merge-depth verification.
- Keep study-specific state in checked-in study records and OPS providers; do
  not reconstruct current posture from scratch artifacts when a status provider
  exists.
