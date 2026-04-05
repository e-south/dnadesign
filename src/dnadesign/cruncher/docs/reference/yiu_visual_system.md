## YIU Visual System

**Audience:** YIU workflow users and maintainers
**Applies to:** `uv run cruncher yiu render|show` and the published BaseRender handoff
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-05

This page owns the named visual directions and information hierarchy for payload-centric YIU views.
Use [YIU Workflow](../guides/yiu_workflow.md) for command flow, [YIU Artifacts](yiu_artifacts.md) for emitted files and render-state semantics, and [Architecture](architecture.md) for module ownership.

### Movement

`bench_strip`

`bench_strip` treats the composite render as a lab-bench strip rather than a poster. The page should read like an operator surface: one dense row for sequence truth, followed by quieter rows for mechanical assembly state. The visual goal is not ornament. It is to make one payload decision legible at a glance without hiding the biology behind decorative chrome.

Space is used asymmetrically. The payload row earns the widest semantic bandwidth because it carries the selected payload, the complement, the mismatch plan, and optional PWM evidence. The split and assembled rows deliberately feel lighter so they behave like procedural confirmation instead of competing stories.

Color stays subordinate to structural meaning. TF-linked palette tokens are allowed to carry motif identity, but the page background, connector treatment, and sequence typography remain restrained. That keeps the PWM layer expressive without letting it overpower the payload truth row.

Scale and rhythm are intentionally stepped. The payload row is compact and information-dense, while the assembly rows preserve larger margins, fewer legends, and calmer connector treatment. The result should feel like descending certainty: selected truth first, then cut geometry, then restored order.

The typography posture is technical rather than editorial. Titles should help a reader orient quickly, but the figure should still be driven by sequence geometry and explicit annotations. Text that does not help navigation or evidence interpretation should stay out of the render.

### Information hierarchy

The hierarchy is intentionally split into evidence, confirmation, and restoration:

- `payload` is the evidence row. It carries the selected payload, mismatch annotations, and PWM overlays when available.
- `split_payload` is the confirmation row. It shows fragment geometry, sticky-end context, and only the lightest necessary labels.
- `assembled_payload` is the restoration row. It returns to payload order and makes the junction span obvious without adding a second story.

Keep that order stable unless the contract changes with it. If the payload row stops being the strongest visual anchor, the page stops being `bench_strip` and starts becoming a generic multi-panel poster.

### Visual translation map

- `payload` uses `evidence_ribbon`: payload truth first, mismatch evidence second, PWM overlays third.
- `split_payload` uses `operator_strip`: fragment geometry and sticky-end context stay centered and legend-light.
- `assembled_payload` uses `operator_strip`: restored payload order and the explicit junction span stay readable without extra ornament.

### Boundary rules

- The producer-side style policy lives in `src/dnadesign/cruncher/src/yiu/visual_system.py`.
- The title policy lives in `src/dnadesign/cruncher/src/yiu/view_styles.py`.
- `cruncher` chooses the named visual direction and style overrides when it publishes bundle views.
- `baserender` consumes those contracts through its public adapter and renderer APIs; it should not infer YIU layout policy from showcase defaults alone.
- Visual-system edits should preserve the information hierarchy unless the contract and docs are updated together.

### Related docs

- [YIU Workflow](../guides/yiu_workflow.md)
- [YIU Artifacts](yiu_artifacts.md)
- [Architecture](architecture.md)
