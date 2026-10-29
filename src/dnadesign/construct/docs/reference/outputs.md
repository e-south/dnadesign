<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Construct docs](../README.md)

# Construct outputs reference

## Output root choices

Packaged workspaces keep their datasets in `outputs/usr_datasets`. This keeps
demo inputs and outputs together. For reuse across workspaces, set an explicit
SeqLedger root in `output.target.root` and the input/template source roots.

Construct does not silently move between these roots. The resolved `input_root` and `output_root` are printed by runtime preflight.

## Dataset write behavior

- output datasets are append-only
- construct fails during preflight if one plan would generate the same output id more than once
- rerunning the same construct into the same dataset with `output.on_conflict=error` fails during preflight
- `output.on_conflict=ignore` reuses existing sequences and completes missing lineage, labels and requested sequence views; conflicting metadata is rejected
- reused sequences count as skipped even when a retry repairs missing metadata
- writing to the same dataset as input is blocked unless `output.allow_same_as_input=true`
- `output.record_source` is optional provenance text only; it does not control the write target

## Lineage columns

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

Pass the same root when inspecting results. For the packaged local demo:

```bash
# Verify the output dataset under the workspace-local root.
seqledger --root outputs/usr_datasets validate \
  anchor_template_slot_a_window_1kb_demo --strict
```

## Records for tracing a run

- config file path
- `spec_id` emitted by `validate --runtime` and `run`
- `construct.workspace.yaml` project inventory
- `construct workspace doctor` for registry/config drift detection
- `inputs/seed_manifest.yaml` for the packaged anchor/template demo
- custom `seed import-manifest` YAML for generic input/template onboarding

## Common output arrangements

- one anchor dataset, one template record, one output dataset
- one candidate dataset, one template record, multiple named input slots from
  each row, one output dataset
- one anchor dataset, one template record, multiple output datasets across multiple workspace projects
- one anchor dataset, one template record, one accumulating output dataset with `output.on_conflict=ignore`
- one anchor dataset, one template record, one accumulating output dataset with `output.on_conflict=error` when distinct projects emit distinct output ids
- workspace-local demo roots or explicit shared SeqLedger roots

Matrix orchestration across multiple templates is expressed as multiple project
entries in the workspace registry. Multiple slots within one template are
first-class runtime schema.

For the cross-tool pattern where multiple construct projects feed one canonical SeqLedger dataset before infer adds derived namespaces, use the shared runbook:
[Construct -> SeqLedger -> Infer shared dataset runbook](../../../seqledger/docs/operations/assembly/construct-infer-shared-dataset-runbook.md).

For the explicit downstream infer handoff fields, see [template-contexts.md](template-contexts.md).
