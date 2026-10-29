## OPAL Test Layout

This test tree is organized by domain to keep cohesion high and reduce search cost.

### Category map

- `cli/`: CLI command behavior, rendering, pressure CLI paths, and workflow command matrix tests.
- `config/`: config loading and schema/registry validation tests.
- `runtime/`: run-loop orchestration, round context, locks, state, and runtime contracts.
- `models/`: model plugin unit tests.
- `objectives/`: objective contracts and objective math/uncertainty behavior.
- `selection/`: selection contracts and selection-method behavior.
- `transforms/`: transforms and y-op behavior.
- `ingest/`: ingest and label-history handling.
- `storage/`: ledger/parquet/store write/read contracts.
- `predict/`: prediction output and inversion behavior.
- `analysis/`: analysis facade and diagnostics behavior.
- `plots/`: plot contract/output tests.
- `notebooks/`: notebook template/autoload smoke tests.
- `platform/`: environment/platform guard tests.

### Placement rules

- Put tests in the smallest domain folder that owns the behavior.
- Keep cross-domain integration tests in `cli/` or `runtime/` unless a clearer home exists.
- Reuse shared helper utilities via `dnadesign.opal.tests._cli_helpers`.
- Prefer adding a new file in an existing domain over creating a new top-level category.
