---
doc_id: construct-template-contexts
title: Construct template contexts
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
---

# Construct template contexts

Use this page when construct is supplying larger resolved contexts for infer or other downstream tools.

### Ownership boundary

- `construct` owns part placement, template realization, and coordinate mapping
- `infer` reads resolved sequences plus `construct__*` metadata
- downstream tools should not reconstruct anchor/template geometry themselves

### Repo-aligned context contract

The general result is a realized sequence plus named part spans and lineage.
For template-backed inference or another downstream analysis, the resolved row
should carry:

- `construct__context_id`
- `construct__context_kind`
- `construct__template_id`
- `construct__anchor_id`
- `construct__anchor_orientation`
- `construct__anchor_start`
- `construct__anchor_end`
- `construct__assembly_mode`
- `construct__slot_count`
- `construct__slots`
- `construct__resolved_length`
- `construct__spec_id`

These values use coordinates on the realized sequence. `construct__slots` is
the complete named-span map. The singular `construct__anchor_*` projection is
the declared focal span for consumers that accept one span; the field name does
not make every Construct part an anchor or impose a biological role.

For multi-slot jobs, `construct__slots` remains authoritative for all parts.
Any job that emits `realized_context` sequence-view variants must therefore
declare a focal handoff span when the view needs `anchor_mean`, either through
`output_variants[].anchor_part`, `realize.focal_part`, or a single part named or
role-tagged `anchor`.
For slot-specific views, prefer `output_variants[].anchor_part`: Construct copies
that named slot's emitted-orientation bounds into the sequence-view
`anchor_start_0` / `anchor_end_0` fields without pretending the whole package has
multiple generic anchors. Downstream Infer configs for those views should use
`bounds_from: sequence_view`.

### What infer expects

`infer` uses those fields to:

- distinguish anchor-only vs template-backed contexts
- compute `anchor_mean` pooling over the anchor token positions inside larger
  emitted contexts
- stamp feature records with stable construct provenance

For causal Evo2 feature extraction, Construct only owns the emitted sequence and
the emitted-orientation span coordinates. Infer still passes the full emitted
context through the model before pooling the anchor span, and a separate
reverse-complement context row is required when downstream analyses need the
complementary causal direction.

Templated infer jobs fail fast when the required `construct__*` fields are missing.

Construct now also fails fast during preflight when a windowed output would clip or wrap the focal anchor so that
`construct__anchor_start` / `construct__anchor_end` cannot be emitted as one contiguous span.
Windowed jobs also fail when `realize.required_slots` names a part that would be
clipped or split in the emitted view.

### Template strategy

The current construct schema remains one-template-per-job. Within that one
template, jobs may assemble multiple named slots from one candidate row.

Use multiple construct projects when you need:

- a 1 kb sequence window
- a plasmid-scale context
- multiple alternative template backbones

That keeps template choice explicit as workspace/config state instead of hiding a template matrix inside one infer job.

### Cross-tool routes

Use one of these next steps after construct materializes template-backed contexts:

- [Construct -> USR -> Infer shared dataset runbook](../../../usr/docs/operations/assembly/construct-infer-shared-dataset-runbook.md): generic shared-dataset handoff into infer and downstream watchers.
- [Promoter characterization feature matrix](../../../usr/docs/operations/promoter/characterization-feature-matrix.md): one study-specific example; it does not define the shared Construct contract.
