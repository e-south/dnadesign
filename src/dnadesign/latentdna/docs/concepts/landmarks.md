# Landmarks

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

Landmarks are explicit named reference sets. They are not inferred from centroids, clusters, or nearest-neighbor heuristics at declaration time.

Current representation modes:

- `rows`
- `centroid`
- `medoid`

Why they matter:

- `distance score` uses them to build reusable control-distance surfaces.
- `enrich score` uses them to summarize neighborhood composition.
- `agreement compare` can use landmark-neighborhood overlap without requiring raw coordinate comparability.

Landmark manifests and downstream distance manifests should always record:

- source dataset or source view
- representation mode
- member count
- metric used

See also:

- [../workflows/landmark-neighborhoods.md](../workflows/landmark-neighborhoods.md)
- [../workflows/control-distances.md](../workflows/control-distances.md)
