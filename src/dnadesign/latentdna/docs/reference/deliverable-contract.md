<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md)

# Deliverable Contract

A deliverable groups the outputs needed to answer one configured analysis question. It names the recipe that builds them, their prerequisites and their acceptance checks.

Status values:

- `ok`
- `attention`
- `missing`
- `error`

Required fields and checks:

- deliverables must declare explicit `title`, `section`, `question`, `summary`,
  `requires`, `outputs`, `docs_refs`, and `acceptance_checks` fields. `kind` and `description` do not supply missing semantic fields.
- `requires` must reference declared config objects or artifact families.
- `outputs` must reference declared objects where relevant and must also be produced by the linked recipe.
- deliverable status evaluates artifact freshness where manifests provide path-backed provenance.
- `deliverable run` delegates to `recipe run` and then recomputes deliverable status from the resulting artifacts.

See also:

- [artifact-manifests.md](artifact-manifests.md)
- [workspace-schema.md](workspace-schema.md)
