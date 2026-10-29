<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Concepts](README.md)

# Coordinate spaces

A vector's coordinates acquire meaning from the model, layer and transformation that produced them. Two vectors with the same number of values are not necessarily comparable coordinate by coordinate.

A view's `coordinate_space_id` declares that relationship. Direct subtraction requires the same declared space, matching dimensions and aligned rows. Assigning the same name to incompatible representations does not make the operation meaningful; the workspace author must establish compatibility.

PCA creates a new coordinate space. To compare new rows with previously reduced rows, apply the same fitted reducer rather than fitting a separate PCA and treating its axes as identical. Normalization also needs an explicit policy because scaling changes the geometry used by distances and neighbors.

For different model spaces, [cross-view agreement](../workflows/cross-view-agreement.md) compares row neighborhoods or clusters instead of raw coordinates. For paired representations in one compatible space, use [context comparisons](../workflows/context-shift.md).
