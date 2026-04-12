# Control Distances

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

This workflow covers landmark distance scoring and scalar margin derivation for control diagnostics.

Primary checked-in promoter-study deliverable:

- `control_distance_margins`

Core artifact path:

1. Materialize the source-backed view that anchors the control distance surface.
2. Score a `distance_set` against declared landmarks such as `spy_p` and `sul_ap`.
3. Derive signed scalar margins with `scalar derive`.
4. Render `distance_scatter` and `distribution` plots from the persisted artifacts.

Key invariants:

- Landmark representations are explicit and recorded in the distance manifest.
- Scalar expressions run through the safe expression surface only.
- Distance and scalar plots never recompute missing prerequisites.

See also:

- [landmark-neighborhoods.md](landmark-neighborhoods.md)
- [scalar-contract.md](../reference/scalar-contract.md)
- [artifact-manifests.md](../reference/artifact-manifests.md)
