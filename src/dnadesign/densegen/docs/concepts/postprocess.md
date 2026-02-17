## Postprocess concepts

This concept page explains how DenseGen finalizes sequences when solved layouts are shorter than the target length. Read it when you need to reason about padding behavior, GC feasibility, and fail-fast semantics.

### Why postprocess exists
This section describes the runtime boundary between solving and final sequence materialization.

DenseGen solves motif-placement constraints first, then applies postprocess pad logic to meet final sequence-length requirements. Postprocess decisions are recorded in runtime artifacts so sequence provenance remains auditable.

### Pad modes
This section explains what each mode allows and how strictness changes failure behavior.

- `off`: no padding is allowed; short sequences fail immediately.
- `strict`: padding is allowed but GC requirements must be met exactly.
- `adaptive`: padding may relax GC bounds when strict targets are infeasible; relaxation is recorded.

### GC feasibility
This section clarifies why short pad regions can make some GC targets impossible.

- Very short pads can have discrete GC outcomes that cannot hit a requested target.
- `strict` treats infeasibility as a hard error.
- `adaptive` can continue with explicit relaxation reporting.

### Config surface
This section points to the keys that control pad behavior.

Use `densegen.postprocess.pad` in config for mode, pad-end selection, GC target/range, and retry bounds. For exact field contracts, use **[config reference](../reference/config.md)**.
