## Construct CLI reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-24

### Command map

- `uv run construct run --config <path> [--dry-run] [--format text|json]`
- `uv run construct compose validate --config <path> [--format text|json]`
- `uv run construct compose run --config <path> [--format text|json]`
- `uv run construct compose review --bundle <artifact-bundle> [--nucleotide-font-size-px <float>] [--format text|json]`
- `uv run construct validate config --config <path> [--runtime] [--format text|json]`
- `uv run construct seed import-manifest --manifest <path> [--root <usr-root>]`
- `uv run construct seed anchor-template-demo [--root <usr-root>] [--manifest <path>]`
- `uv run construct workspace where [--root <workspace-root>] [--profile <profile>]`
- `uv run construct workspace init --id <workspace-id> [--root <workspace-root>] [--profile <profile>]`
- `uv run construct workspace show --workspace <workspace-dir> [--format text|json]`
- `uv run construct workspace doctor --workspace <workspace-dir> [--format text|json]`
- `uv run construct workspace validate-project --workspace <workspace-dir> --project <id> [--runtime] [--format text|json]`
- `uv run construct workspace run-project --workspace <workspace-dir> --project <id> [--dry-run] [--format text|json]`

### `validate config`

Use `validate config` before `run`. With `--runtime`, the command resolves:

- input dataset/root
- template source, record id, and SHA-256
- realization mode, `realize.window` contract, and placement contract
- spec fingerprint (`spec_id`)
- projected output ids and lengths
- existing output collisions according to `output.on_conflict`

`--format json` emits the same preflight contract as machine-readable JSON so
agents or downstream automation can inspect placement strategy, guard posture,
planned rows, and template provenance without scraping text output.

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

`--format json` emits the run summary as a machine-readable payload, including
the dry-run flag, output root, and spec id.

### `compose`

`compose` is the workspace-less route for declared linear ssDNA segment specs.
Use it when the caller has already selected the parts and wants Construct to
emit a local artifact bundle rather than a USR-backed template/context run.

- `compose validate` parses `linear_ssdna_composition_v1`, checks segment
  transforms, span bounds, repeat policy, and planned sequence length, and
  exits before writing outputs.
- `compose run` writes the artifact bundle declared by the config, including
  assembled sequence exports, manifest metadata, visual contracts, and optional
  Folding/BaseRender handoff artifacts.
- `compose review` reads an existing bundle and publishes the two-panel
  composition review SVG plus high-resolution PNG sibling when the required
  structure and component-span artifacts are present.

Failure posture:

- invalid YAML or schema fails before runtime work
- annotations cannot create sequence and may only interpret validated spans
- visual and folding evidence are generated for the canonical component unit
  unless a future contract explicitly declares a repeat-expanded evidence
  surface
- Folding remains advisory or required according to the composition config, but
  missing backend state is always explicit

### `seed anchor-template-demo`

This command bootstraps the packaged anchor/template demo inputs:

- `anchor_parts_demo`
- `template_parts_demo`

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
- `workspace list`: show local workspaces in the active root first, then packaged templates that have not been copied locally
- `workspace show`: read `construct.workspace.yaml` and print the workspace project inventory
- `workspace doctor`: fail if any workspace project entry drifts from its config file or points at a missing config
- `workspace validate-project`: resolve one project by registry id and run the same validation surface as `validate config`
- `workspace run-project`: resolve one project by registry id and run the same execution surface as `run`

`workspace show`, `workspace doctor`, `workspace validate-project`, and
`workspace run-project` all support `--format json` for harness-friendly
inspection and preflight gating.

Workspace registry contract:

- every construct workspace should carry `construct.workspace.yaml`
- the default workspace root is the current working directory; use `--root` or `CONSTRUCT_WORKSPACE_ROOT` to override it
- each project entry tracks one config artifact under `project.artifacts.config` and one intended input/template/output contract under `project.contract`
- `workspace doctor` rejects both config path drift and config job-id drift before execution
- `workspace show` no longer carries a descriptive `flow` string; the runtime contract lives in the config itself and the audited routing contract lives in `project.contract`
- multi-template studies are represented as multiple project entries, not multiple templates inside one construct job
- packaged profiles currently include `anchor-template-demo` and `anchor-template-shared-dataset-demo`
