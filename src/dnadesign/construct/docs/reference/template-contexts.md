## Construct Template Contexts

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-18

Use this page when construct is supplying larger resolved contexts for infer or other downstream tools.

### Ownership boundary

- `construct` owns anchor placement, template realization, and coordinate mapping
- `infer` reads resolved sequences plus `construct__*` metadata
- downstream tools should not reconstruct anchor/template geometry themselves

### Repo-aligned context contract

For template-backed inference or other downstream analysis, the resolved construct row should carry:

- `construct__context_id`
- `construct__context_kind`
- `construct__template_id`
- `construct__anchor_id`
- `construct__anchor_orientation`
- `construct__anchor_start`
- `construct__anchor_end`
- `construct__resolved_length`
- `construct__spec_id`

These values are emitted relative to the realized sequence that construct writes.

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

### Template strategy

The current construct schema remains one-template-per-job.

Use multiple construct projects when you need:

- a 1 kb sequence window
- a plasmid-scale context
- multiple alternative template backbones

That keeps template choice explicit as workspace/config state instead of hiding a template matrix inside one infer job.

### Cross-tool route

Use one of these next steps after construct materializes template-backed contexts:

- [Construct -> USR -> Infer shared dataset runbook](../../../usr/docs/operations/assembly/construct-infer-shared-dataset-runbook.md): generic shared-dataset handoff into infer and downstream watchers.
- [Promoter characterization feature matrix](../../../usr/docs/operations/promoter/characterization-feature-matrix.md): promoter-study branch for feature extraction, cluster, and OPAL prep.
