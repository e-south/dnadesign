# Deliverable Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-11

Deliverables are semantic bundles over one linked recipe plus declared prerequisites and outputs.

Current status vocabulary:

- `ok`
- `attention`
- `missing`
- `error`

Important current behavior:

- deliverables must declare explicit `title`, `section`, `question`, `summary`,
  `requires`, `outputs`, `docs_refs`, and `acceptance_checks` fields; legacy
  `kind` and `description` fallbacks are rejected.
- `requires` must reference declared config objects or artifact families.
- `outputs` must reference declared objects where relevant and must also be produced by the linked recipe.
- deliverable status evaluates artifact freshness where manifests provide path-backed provenance.
- `deliverable run` delegates to `recipe run` and then recomputes deliverable status from the resulting artifacts.

See also:

- [artifact-manifests.md](artifact-manifests.md)
- [workspace-schema.md](workspace-schema.md)
