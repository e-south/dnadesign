![Studies banner](assets/studies-banner.svg)

`studies` contains narrow helpers for checked-in study records that need code without becoming generic tool features. Ops owns the status and preflight APIs; this package keeps concrete study logic inside the owning study package and shared record parsing under `core`.

## Documentation

- [Study records index](../../../docs/studies/README.md): checked-in study
  manifests, routes, and status notes.
- [Ops README](../ops/README.md): status, preflight, and orchestration entry
  points.
- [Repository docs index](../../../docs/README.md): cross-tool workflow routing.

## Source Layout

- `assets/`: study-package visual/static assets.
- `core/`: study-record contracts, loaders, selectors, and preflight planning
  primitives that are not tied to one study.
- `studies/<study-id>/`: concrete study packages. Study-specific status,
  preflight, compiler, or handoff logic stays inside the owning study.
- `tests/`: package tests outside concrete study packages.

Do not add a generic cross-study status layer for one study's behavior. Extract
shared code only after a second real study proves the contract.
