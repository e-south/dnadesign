<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Reference](README.md)

# Artifact naming grammar

Choose IDs that describe the represented data or comparison. Include the model, layer, sequence context or transformation when those distinctions matter within the workspace.

Examples:

- `encoder_layer12_insert_mean`
- `encoder_layer12_context_mean`
- `insert_context_distances`
- `reference_neighborhoods`
- `representation_health`

Use the same ID when referring to an artifact from recipes, exports and plots. Prefer names with a stable scientific or computational meaning over publication panel numbers or temporary priorities such as `main`.

Keep display labels and layout in configuration. An ID identifies an artifact; it does not determine model identity, plot colors or panel arrangement. See [plot presentation](workspace-schema.md#plot-presentation) and [manifests](artifact-manifests.md).
