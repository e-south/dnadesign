![Studies banner](assets/studies-banner.svg)

`studies` contains shared study-record mechanics and narrow helpers for
deliberately public study units. Private study code and records belong in an
external private workspace that imports dnadesign's public APIs. Ops owns the
status and preflight APIs; `core` owns neutral record parsing.

## Documentation

- [Study records index](../../../docs/studies/README.md): checked-in study
  manifests, routes, and status notes.
- [Public/private boundary](../../../docs/studies/reference/public-private-boundary.md): decide whether a study may live in this public repository.
- [Ops README](../ops/README.md): status, preflight, and orchestration entry
  points.
- [Repository docs index](../../../docs/README.md): cross-tool workflow routing.

## Source Layout

- `assets/`: study-package visual/static assets.
- `core/`: study-record contracts, loaders, selectors, and preflight planning
  primitives that are not tied to one study.
- `units/<study-id>/`: concrete study source units. Study-specific status,
  preflight, compiler, or handoff logic stays inside the owning study.
- `units/<study-id>/tests/`: concrete study tests. Study-specific tests stay
  inside the owning unit instead of a parallel package-level test tree.
- `tests/test_*.py`: shared `studies.core` and package-level contract tests
  only. Do not put concrete study tests at the package-level test root.

Do not add a generic cross-study status layer for one study's behavior. Extract
shared code only after a second real study proves the contract.
