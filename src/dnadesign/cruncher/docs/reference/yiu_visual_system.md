## YIU Visual System

**Audience:** YIU workflow users and maintainers
**Applies to:** `uv run cruncher yiu render|show` and the published BaseRender handoff
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-05
**Last updated by:** cruncher-maintainers on 2026-04-05

YIU renders one composite page with a clear priority order: payload evidence first, split geometry second, and assembled confirmation third. The named visual system for that page is `bench_strip`.

Use [YIU Workflow](../guides/yiu_workflow.md) for command flow, [YIU Artifacts](yiu_artifacts.md) for emitted files and render-state semantics, and [Architecture](architecture.md) for module ownership.

<!-- docs:toc:off -->

### Quick map

- `payload` uses `evidence_ribbon`
- `split_payload` uses `operator_strip`
- `assembled_payload` uses `operator_strip`

### `bench_strip`

`bench_strip` treats the composite render as an operator surface rather than a poster. One dense row carries sequence truth, mismatch evidence, and optional PWM overlays. The rows below it confirm how the payload is split and then reassembled.

The payload row gets the most bandwidth because it carries the selected payload, the complement, the mismatch plan, and motif context when PWM is effective. The split and assembled rows stay lighter so they confirm geometry without competing with the payload row.

Color stays subordinate to structure. TF-linked palette tokens can identify motifs, but the page background, connectors, and sequence typography stay restrained so the evidence row remains readable.

Scale is intentionally stepped: selected truth first, cut geometry second, restored order third.

### Information hierarchy

The hierarchy is split into evidence, confirmation, and restoration:

- `payload` is the evidence row. It carries the selected payload, mismatch annotations, and PWM overlays when available.
- `split_payload` is the confirmation row. It shows fragment geometry, sticky-end context, and only the lightest necessary labels.
- `assembled_payload` is the restoration row. It returns to payload order and makes the junction span obvious without adding a second story.

Keep that order stable unless the contract changes with it. If the payload row stops being the strongest visual anchor, the render is no longer `bench_strip`.

### Visual translation map

- `payload` uses `evidence_ribbon`: payload truth first, mismatch evidence second, PWM overlays third.
- `split_payload` uses `operator_strip`: fragment geometry and sticky-end context stay centered and legend-light.
- `assembled_payload` uses `operator_strip`: restored payload order and the explicit junction span stay readable without extra ornament.

All three sequence-evidence families publish the optimizer-chosen junction through shared `meta.span_backdrops` metadata so BaseRender can draw the same rounded light-blue duplex backdrop behind the active span without introducing a YIU-only renderer path.

### Boundary rules

- [Architecture](architecture.md) owns the full module map.
- The producer-owned style seed lives in `src/dnadesign/cruncher/src/yiu/visual_foundations.py`.
- Named direction deltas live in `src/dnadesign/cruncher/src/yiu/visual_directions.py`.
- The view registry and style profiles live in `src/dnadesign/cruncher/src/yiu/visual_system.py`.
- `evidence_ribbon` and `operator_strip` should share the same `bench_strip` foundation and diverge only where emphasis changes.
- The title policy lives in `src/dnadesign/cruncher/src/yiu/view_styles.py`.
- `cruncher` chooses the named visual direction and style overrides when it publishes bundle views.
- `baserender` consumes those contracts through its public adapter and renderer APIs; it should not infer YIU layout policy from consumer-side showcase defaults or hidden heuristics.
- Visual-system edits should preserve the information hierarchy unless the contract and docs are updated together.

### Related docs

- [YIU Workflow](../guides/yiu_workflow.md)
- [YIU Artifacts](yiu_artifacts.md)
- [Architecture](architecture.md)
