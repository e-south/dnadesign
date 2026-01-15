# DenseGen Alignment Audit (Refactor Overhaul)

This audit tracks alignment between the implementation, documentation, and the
developer spec (`dev/developer-spec.md`). It is intentionally concise and
focused on **pragmatic programming**: explicit contracts, decoupled modules,
and deterministic behavior.

As of **2026-01-15**.

## In-sync highlights

- **Solver strands**: `solver.strands` (`single|double`) is implemented, validated,
  and recorded in metadata (`densegen__solver_strands`).
- **Regulator constraints**:
  - `required_regulators`, `min_required_regulators`, `min_count_by_regulator`
    are enforced via dense-arrays for exact strategies.
  - `approximate` runs are validated in the pipeline for consistent behavior.
- **Regulator labels** are carried explicitly (no dependency on parsing `tf:tfbs` strings).
- **Metadata registry** includes the new constraint fields and is enforced in Parquet.
- **Docs** updated for config, generation constraints, and metadata outputs.
- **Run-scoped I/O**: configs live inside `densegen.run.root`; outputs/logs/plots are confined to the run root.

## Remaining gaps (per developer spec)

- **USR parity test**: when USR is installed, add a test asserting DenseGen IDs
  match USR IDs for the same sequences.

## Pragmatic programming checks

- **Single source of truth**: `config/` and `core/metadata_schema.py` remain canonical.
- **Explicit policies**: solver/sampling/gap-fill policies are written into metadata.
- **No fallbacks**: invalid configs or infeasible constraints fail fast with diagnostics.

## Follow-ups (optional)

- Add a small integration test covering `solver.strands` for a known toy library.
- Gate regulator constraints on dense-arrays capability with a clear error if missing.
