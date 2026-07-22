## Stress OPS Catalog

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This directory indexes OPS catalog pages for the stress/ethanol/cipro growth
study. The root study record stays the entrypoint; catalog contracts live under
`contracts/`, and runbook-catalog sidecars live under `contracts/registry/`.

- [Status contract](contracts/status.md): read-only study snapshot.
- [Preflight contract](contracts/preflight.md): read-only command and path readiness.
- `contracts/registry/*.registry.yaml`: runbook-catalog sidecars for the
  adjacent contract docs.
