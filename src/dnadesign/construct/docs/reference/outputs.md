## construct outputs reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14

### Output root choices

Two patterns are supported:

1. Workspace-local USR root
- Default packaged-workspace pattern: `outputs/usr_datasets`
- Best for study-scoped provenance, isolation, and replay

2. Explicit shared USR root
- Use `output.root` plus matching input/template roots when a shared dataset mirror is intentional
- Best for curated shared datasets or downstream cross-workspace reuse

Construct does not silently move between these roots. The resolved `input_root` and `output_root` are printed by runtime preflight.

### Dataset write behavior

- output datasets are append-only
- construct fails during preflight if one plan would generate the same output id more than once
- rerunning the same construct into the same dataset with `output.on_conflict=error` fails during preflight
- `output.on_conflict=ignore` skips already-present output ids and reports the skipped count
- writing to the same dataset as input is blocked unless `output.allow_same_as_input=true`

### Lineage columns

Construct writes standardized `construct__*` lineage columns, including:

- job and spec fingerprint
- template identity, source, and checksum
- input dataset/field provenance
- focal window coordinates and full construct length
- part count, execution order, realized coordinates, and template coordinates

Use `uv run usr head <dataset>` or `uv run usr validate <dataset> --strict` to inspect or verify the resulting records.

### Provenance surfaces

- config file path
- `spec_id` emitted by `validate --runtime` and `run`
- `construct.workspace.yaml` project inventory
- `construct workspace doctor` for registry/config drift detection
- `inputs/seed_manifest.yaml` for the packaged promoter-swap demo
- custom `seed import-manifest` YAML for generic input/template onboarding

### Pragmatic flow patterns supported now

- one anchor dataset, one template record, one output dataset
- one anchor dataset, one template record, multiple output datasets across multiple workspace projects
- one anchor dataset, one template record, one accumulating output dataset with `output.on_conflict=ignore`
- workspace-local demo roots or explicit shared USR roots

Matrix orchestration across multiple templates or slots is currently expressed as multiple project entries in the workspace registry, not a multi-template runtime schema.
