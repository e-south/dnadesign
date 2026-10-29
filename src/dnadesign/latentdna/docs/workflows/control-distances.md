<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Workflows](README.md)

# Compare sequences with references

Reference distances locate each row relative to named sequences or groups in the same representation space. A landmark can retain its member rows, use their centroid, or select a medoid; declare that choice before scoring.

## Build the comparison

1. Materialize the view containing the query and reference rows.
2. Declare landmark membership with metadata predicates and choose its representation.
3. Run `distance score` with the view, landmarks and distance metric.
4. If needed, derive a scalar contrast from the saved distances and render it as a scatter or distribution.

For a workspace with an existing `embeddings` view and a declared `reference` landmark:

```bash
# Save cosine distances to the configured reference.
latentdna distance score reference_distances \
  --workspace /path/to/workspace --view embeddings \
  --landmark reference --metric cosine
```

For cosine distance, smaller values indicate greater directional resemblance. If a scalar subtracts two distances, state the subtraction order: changing it reverses the sign. A reference-relative score describes the chosen representation and reference membership; its relationship to measured function needs separate evidence.

The distance manifest records the reference representation and metric. Plotting reads the persisted result and does not create missing prerequisites. See [landmarks](../concepts/landmarks.md), [scalar tables](../reference/scalar-contract.md) and [manifests](../reference/artifact-manifests.md).
