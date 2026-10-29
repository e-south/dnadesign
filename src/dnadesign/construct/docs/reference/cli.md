<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Construct docs](../README.md)

# Construct CLI reference

## Command map

The forms below show command syntax; replace angle-bracket placeholders and
omit square-bracketed optional arguments when they are not needed. For a
runnable sequence, use [getting started](../getting-started.md).

- `construct run --config <path> [--seqledger-root <absolute-path>] [--dry-run] [--format text|json]`
- `construct compose validate --config <path> [--format text|json]`
- `construct compose run --config <path> [--format text|json]`
- `construct compose review --bundle <artifact-bundle> [--nucleotide-font-size-px <float>] [--format text|json]`
- `construct validate config --config <path> [--seqledger-root <absolute-path>] [--runtime] [--format text|json]`
- `construct seed import-manifest --manifest <path> [--root <seqledger-root>]`
- `construct seed anchor-template-demo [--root <seqledger-root>] [--manifest <path>]`
- `construct workspace list [--root <workspace-root>]`
- `construct workspace where [--root <workspace-root>] [--profile <profile>]`
- `construct workspace init --id <workspace-id> [--root <workspace-root>] [--profile <profile>]`
- `construct workspace show --workspace <workspace-dir> [--format text|json]`
- `construct workspace doctor --workspace <workspace-dir> [--format text|json]`
- `construct workspace validate-project --workspace <workspace-dir> --project <id> [--seqledger-root <absolute-path>] [--runtime] [--format text|json]`
- `construct workspace run-project --workspace <workspace-dir> --project <id> [--seqledger-root <absolute-path>] [--dry-run] [--format text|json]`

`--seqledger-root` selects the same storage location for validation, dry-run and saved outputs.
It must name an absolute, existing directory that is not a symbolic link. When
present, it supplies the input, template, and output SeqLedger root; the command does
not return to roots declared in the job config between preflight and execution.

## `validate config`

Use `validate config` before `run`. With `--runtime`, the command resolves:

- input dataset/root
- template source, record id, and SHA-256
- realization mode, `realize.window` contract, and placement contract
- spec fingerprint (`spec_id`)
- projected output ids and lengths
- existing output collisions according to `output.on_conflict`

`--format json` emits the same preflight contract as machine-readable JSON so
scripts can inspect placement strategy, placement guards,
planned rows, and template provenance without scraping text output.

Validation failures:

- invalid YAML or schema: fail before runtime work
- missing datasets/records/fields: fail before output planning
- duplicate planned output ids inside one run: fail during runtime preflight before any write
- output-id collisions with `output.on_conflict=error`: fail during runtime preflight, not after a partial write
- input and output resolving to the same dataset/root: fail unless `output.allow_same_as_input=true`

## `run`

`run` realizes sequences and writes them into the configured output dataset.

- default policy is append-only with `output.on_conflict=error`
- `output.on_conflict=ignore` reuses existing sequences and completes missing
  lineage, derived metadata, labels, and requested sequence views on retry.
  Conflicting metadata is rejected rather than overwritten. Distinct sequence
  views may record another derivation without replacing the sequence row's
  original provenance.
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
Reused sequence IDs count as skipped even when their missing metadata is repaired.
Success means all requested metadata and views have been persisted; a failed
run may have committed some stages and can be retried with `ignore`.

## `compose`

`compose` is the workspace-less route for declared linear ssDNA segment specs.
Use it when the caller has already selected the parts and wants Construct to
emit a local artifact bundle rather than a SeqLedger-backed template/context run.

- `compose validate` parses `linear_ssdna_composition_v1`, checks segment
  transforms, span bounds, repeat policy, and planned sequence length, and
  exits before writing outputs.
- `compose run` writes the artifact bundle declared by the config, including
  assembled sequence exports, manifest metadata, visual contracts, and optional
  Folding/BaseRender handoff artifacts.
- `compose review` reads an existing bundle and publishes the two-panel
  composition review SVG plus high-resolution PNG sibling when the required
  structure and component-span artifacts are present.

Validation failures:

- invalid YAML or schema fails before runtime work
- annotations cannot create sequence and may only interpret validated spans
- visual and folding evidence are generated for the canonical component unit

- Folding remains advisory or required according to the composition config, but
  missing backend state is always explicit

## `seed anchor-template-demo`

This command bootstraps the packaged anchor/template demo inputs:

- `anchor_parts_demo`
- `template_parts_demo`

It also writes:

- `construct_seed__*` provenance overlays
- `usr_label__primary` / `usr_label__aliases`
- an optional manifest with record ids and slot coordinates

Pass `--root` when importing or seeding data, for example
`--root outputs/usr_datasets` inside a workspace. The seed command also accepts
`DNADESIGN_SEQLEDGER_ROOT`; an explicit `--root` takes precedence.

## `seed import-manifest`

Import your own DNA parts or templates from a [sequence manifest](seed-manifest.md).

- one manifest can create one or more datasets
- dataset ids stay biological and semantic at the SeqLedger layer
- record labels go into `usr_label__primary` / `usr_label__aliases`
- construct bootstrap provenance goes into `construct_seed__*`
- duplicate sequences stay idempotent because import uses append-only `on_conflict=ignore`

## `workspace` commands

- `workspace where`: show workspace root resolution plus packaged profile source
- `workspace init`: scaffold a blank workspace or copy a packaged profile
- `workspace list`: show local workspaces in the active root first, then packaged templates that have not been copied locally
- `workspace show`: read `construct.workspace.yaml` and print the workspace project inventory
- `workspace doctor`: fail if any workspace project entry drifts from its config file or points at a missing config
- `workspace validate-project`: resolve one project by registry id and run the same validation surface as `validate config`
- `workspace run-project`: resolve one project by registry id and run the same execution surface as `run`

`workspace show`, `workspace doctor`, `workspace validate-project`, and
`workspace run-project` all support `--format json` for scripts and automated validation.

Workspace registry contract:

- every construct workspace should carry `construct.workspace.yaml`
- the default workspace root is the current working directory; use `--root` or `CONSTRUCT_WORKSPACE_ROOT` to override it
- each project entry tracks one config artifact under `project.artifacts.config` and one intended input/template/output contract under `project.contract`
- `workspace doctor` rejects both config path drift and config job-id drift before execution
- `workspace show` lists project configurations and expected dataset identities; `workspace doctor` checks that they agree
- multi-template studies are represented as multiple project entries, not multiple templates inside one construct job
- packaged profiles currently include `anchor-template-demo` and `anchor-template-shared-dataset-demo`
