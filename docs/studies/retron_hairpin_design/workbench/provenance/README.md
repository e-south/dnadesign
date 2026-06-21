## Retron Workbench Provenance

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-20

This lane records what was run against a workbench design set. It stores compact
manifests, command contracts, hashes, and output posture, not bulky generated
artifacts.

### Route By Record Type

| Need | Open |
| --- | --- |
| Catalog compile invocation and digest | [compiler_runs/](compiler_runs/README.md) |
| GenBank, plot, or sequence-bundle materialization posture | [materializations/](materializations/README.md) |

### Boundary

Run records cite design sets and compiler inputs. They do not replace
`../design_sets/` as the source of experimental meaning, and they do not replace
generated output bundles.
