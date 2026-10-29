<!--
---
doc_id: construct-template-contexts
title: Construct template contexts
owner: dnadesign-maintainers
status: active
last_verified: 2026-10-28
---

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Construct docs](../README.md)

# Construct template contexts

Use these output fields to analyze an inserted part within its surrounding
template sequence. Construct saves the assembled sequence and part coordinates;
Infer and other tools can read those coordinates directly.

## Sequence and coordinate fields

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
`anchor_start_0` / `anchor_end_0` fields for that slot. Infer configs for those
views should use `bounds_from: sequence_view`.

## What Infer expects

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

Construct fails during preflight when a windowed output would clip or wrap the focal anchor so that
`construct__anchor_start` / `construct__anchor_end` cannot be emitted as one contiguous span.
Windowed jobs also fail when `realize.required_slots` names a part that would be
clipped or split in the emitted view.

## Template strategy

Each template-realization job uses one template. Within that one
template, jobs may assemble multiple named slots from one candidate row.

Use separate projects to compare:

- a 1 kb sequence window
- a plasmid-scale context
- multiple alternative template backbones

Each configuration then names its template and window settings explicitly.

## Cross-tool routes

After saving the assembled sequences:

- [Pass a shared dataset to Infer](../../../seqledger/docs/operations/assembly/construct-infer-shared-dataset-runbook.md) for sequence-model calculations.
- [Explore the promoter characterization example](../../../seqledger/docs/operations/promoter/characterization-feature-matrix.md) for a worked analysis.
