## Construct outputs reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-15

### Output root choices

Two patterns are supported:

1. Workspace-local USR root
- Default packaged-workspace pattern: `outputs/usr_datasets`
- Best for study-scoped provenance, isolation, and replay

2. Explicit shared USR root
- Use `output.target.root` plus matching input/template source roots when a shared dataset mirror is intentional
- Best for curated shared datasets or downstream cross-workspace reuse

Construct does not silently move between these roots. The resolved `input_root` and `output_root` are printed by runtime preflight.

### Dataset write behavior

- output datasets are append-only
- construct fails during preflight if one plan would generate the same output id more than once
- rerunning the same construct into the same dataset with `output.on_conflict=error` fails during preflight
- `output.on_conflict=ignore` skips already-present output ids and reports the skipped count
- writing to the same dataset as input is blocked unless `output.allow_same_as_input=true`
- `output.record_source` is optional provenance text only; it does not control the write target

### Lineage columns

Construct writes standardized `construct__*` lineage columns, including:

- job and spec fingerprint
- template identity, source, and checksum
- input dataset plus input record provenance
- focal-part length plus resolved window semantics, bounds, and emitted geometry
- one compact `construct__parts` column with execution order, placement kind, orientation, realized coordinates, and template coordinates
- `construct__assembly_mode`, `construct__slot_count`, and
  `construct__slots` for named-slot assembly, including slot role, sequence
  field, forward span, emitted-orientation span, placement kind, and template
  span
- template-context fields such as `construct__context_id`, `construct__anchor_id`, `construct__anchor_start`, `construct__anchor_end`, and `construct__resolved_length`

For template-backed downstream handoffs, construct rejects window configs that clip or split the focal anchor, because
those runs cannot emit valid `construct__anchor_start` / `construct__anchor_end` coordinates.
Jobs may also declare `realize.required_slots`; construct rejects any window
that would clip or split one of those named slots.
For multi-slot representation work, output variants can also declare
`anchor_part`. That writes slot-specific sequence-view bounds while preserving a
single base sequence row and the complete `construct__slots` audit trail.

When the input dataset already carries `usr_label__primary` / `usr_label__aliases`, construct carries those labels onto the derived output rows as the analyst-facing source names. Those labels are convenience labels, not uniqueness guarantees for derived construct outputs; use `construct__*` lineage to disambiguate source/template/window context.

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
- one candidate dataset, one template record, multiple named input slots from
  each row, one output dataset
- one anchor dataset, one template record, multiple output datasets across multiple workspace projects
- one anchor dataset, one template record, one accumulating output dataset with `output.on_conflict=ignore`
- one anchor dataset, one template record, one accumulating output dataset with `output.on_conflict=error` when distinct projects emit distinct output ids
- workspace-local demo roots or explicit shared USR roots

Matrix orchestration across multiple templates is expressed as multiple project
entries in the workspace registry. Multiple slots within one template are
first-class runtime schema.

For the cross-tool pattern where multiple construct projects feed one canonical USR dataset before infer adds derived namespaces, use the shared runbook:
[Construct -> USR -> Infer shared dataset runbook](../../../usr/docs/operations/assembly/construct-infer-shared-dataset-runbook.md).

For the explicit downstream infer handoff fields, see [template-contexts.md](template-contexts.md).
