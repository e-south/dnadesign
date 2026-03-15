## construct CLI reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14

### Command map

- `uv run construct run --config <path> [--dry-run]`
- `uv run construct validate config --config <path> [--runtime]`
- `uv run construct seed import-manifest --manifest <path> [--root <usr-root>]`
- `uv run construct seed promoter-swap-demo [--root <usr-root>] [--manifest <path>]`
- `uv run construct workspace where [--root <workspace-root>] [--profile <profile>]`
- `uv run construct workspace init --id <workspace-id> [--root <workspace-root>] [--profile <profile>]`
- `uv run construct workspace show --workspace <workspace-dir>`
- `uv run construct workspace doctor --workspace <workspace-dir>`
- `uv run construct workspace validate-project --workspace <workspace-dir> --project <id> [--runtime]`
- `uv run construct workspace run-project --workspace <workspace-dir> --project <id> [--dry-run]`

### `validate config`

Use `validate config` before `run`. With `--runtime`, the command resolves:

- input dataset/root
- template source, record id, and SHA-256
- realization mode, focal settings, and placement contract
- spec fingerprint (`spec_id`)
- projected output ids and lengths
- existing output collisions according to `output.on_conflict`

Failure posture:

- invalid YAML or schema: fail before runtime work
- missing datasets/records/fields: fail before output planning
- duplicate planned output ids inside one run: fail during runtime preflight before any write
- output-id collisions with `output.on_conflict=error`: fail during runtime preflight, not after a partial write
- input and output resolving to the same dataset/root: fail unless `output.allow_same_as_input=true`

### `run`

`run` realizes sequences and writes them into the configured output dataset.

- default policy is append-only with `output.on_conflict=error`
- `output.on_conflict=ignore` keeps runs idempotent by skipping already-present output ids
- `--dry-run` performs the same planning path without writing data

Run output reports:

- `rows_planned`
- `rows_written`
- `rows_skipped_existing`
- `output_root`
- `output_dataset`
- `spec_id`

### `seed promoter-swap-demo`

This command bootstraps the packaged promoter-swap demo inputs:

- `mg1655_promoters`
- `plasmids`

It also writes:

- `construct_seed__*` provenance overlays
- `usr_label__primary` / `usr_label__aliases`
- an optional manifest with record ids and slot coordinates

Standalone `construct seed` defaults to the canonical repo USR root when `--root` is omitted. Packaged workspaces should pass an explicit workspace-local root, typically `outputs/usr_datasets`.

### `seed import-manifest`

Use this when you have your own anchors or templates and want construct to materialize them into USR without hand-editing datasets.

- one manifest can create one or more datasets
- dataset ids stay biological and semantic at the USR layer
- record labels go into `usr_label__primary` / `usr_label__aliases`
- construct bootstrap provenance goes into `construct_seed__*`
- duplicate sequences stay idempotent because import uses append-only `on_conflict=ignore`

### `workspace` commands

- `workspace where`: show workspace root resolution plus packaged profile source
- `workspace init`: scaffold a blank workspace or copy a packaged profile
- `workspace show`: read `construct.workspace.yaml` and print the workspace project inventory
- `workspace doctor`: fail if any workspace project entry drifts from its config file or points at a missing config
- `workspace validate-project`: resolve one project by registry id and run the same validation surface as `validate config`
- `workspace run-project`: resolve one project by registry id and run the same execution surface as `run`

Workspace registry contract:

- every construct workspace should carry `construct.workspace.yaml`
- each project entry maps one config file to its intended input/template/output contract
- multi-template studies are represented as multiple project entries, not multiple templates inside one construct job
