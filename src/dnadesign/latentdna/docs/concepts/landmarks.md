<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Concepts](README.md)

# Landmarks

A landmark is a named set of reference rows selected explicitly from a source or view. Membership comes from configuration, not from a cluster label or a nearest-neighbor guess.

Choose how the members represent the reference:

| Mode | Reference used |
| --- | --- |
| `rows` | The selected member vectors |
| `centroid` | Their mean vector |
| `medoid` | A representative member of the set |

These choices answer different questions. Distances to individual references preserve differences between members; a centroid summarizes their average location. A medoid remains an observed member rather than an average vector.

[Reference-distance analysis](../workflows/control-distances.md) scores query rows against these declarations. [Neighborhood enrichment](../workflows/landmark-neighborhoods.md) summarizes nearby cohort membership, and agreement can compare those neighborhoods across views. Manifests record source, representation mode, member count and metric so the reference can be interpreted with the result.
