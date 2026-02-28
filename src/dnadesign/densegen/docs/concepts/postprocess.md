## Postprocessing model

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28
This concept page explains how DenseGen finalizes sequences when solved layouts are shorter than the target length. Read it when you need to reason about padding behavior, GC feasibility, and fail-fast semantics.

### Why postprocessing exists

DenseGen solves motif-placement constraints first, then applies postprocess pad logic to meet final sequence-length requirements. Postprocess decisions are recorded in runtime artifacts so sequence provenance remains auditable.

### Pad modes

- `off`: no padding is allowed; short sequences fail immediately.
- `strict`: padding is allowed but GC requirements must be met exactly.
- `adaptive`: padding may relax GC bounds when strict targets are infeasible; relaxation is recorded.

### GC feasibility

- Very short pads can have discrete GC outcomes that cannot hit a requested target.
- `strict` treats infeasibility as a hard error.
- `adaptive` can continue with explicit relaxation reporting.

### Configuration keys

Use `densegen.postprocess.pad` in config for mode, pad-end selection, GC target/range, and retry bounds. For exact field contracts, use **[config reference](../reference/config.md)**. For how postprocess interacts with solve constraints, use **[generation model](generation.md)**.
