## YIU Workflow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-04

YIU is a payload-centric rendering workflow with a strict v4 contract.
Use this guide for command flow and operator posture. Use [YIU Spec Reference](../reference/yiu_spec.md) for input and normalization rules, [YIU Artifacts](../reference/yiu_artifacts.md) for emitted files and render-state semantics, [YIU Visual System](../reference/yiu_visual_system.md) for named visual directions and hierarchy, and [Cruncher architecture](../reference/architecture.md) for module ownership.

The public lane is:

`input payload -> normalized payload -> optimized junction/mismatch plan -> canonical bundle -> BaseRender`

### Documentation ownership

- Use [YIU Workspace Demo](../demos/demo_yiu_workspace.md) for the checked-in workspace and runbook.
- Use this guide for command flow and operator posture.
- Use [YIU Visual System](../reference/yiu_visual_system.md) for the `bench_strip` design language and the `evidence_ribbon` versus `operator_strip` split across the three views.
- Use [YIU Spec Reference](../reference/yiu_spec.md) for schema and normalization only.
- Use [YIU Artifacts](../reference/yiu_artifacts.md) for emitted files, render-status semantics, and the shared inspection surface.
- Use [Cruncher architecture](../reference/architecture.md) when you need module ownership or app/domain boundaries.

### Start here

Use the shortest path that matches your job:

1. [YIU Workspace Demo](../demos/demo_yiu_workspace.md) for the checked-in workspace and machine runbook.
2. `cruncher yiu validate` when you only need schema and payload-plan verification.
3. `cruncher yiu render --emit-renders` when you need the canonical bundle plus the composite operator PDF.
4. `cruncher yiu show` when you need one fail-fast inspection surface for manifest, inventory, payload, and render integrity.

The checked-in reference workspace lives at `src/dnadesign/cruncher/workspaces/demo_yiu_payload`.

YIU accepts two first-class inputs:

- `user_sequence`
- `sample_hit`

Both inputs normalize into one payload object and publish exactly three views:

- `payload`
- `split_payload`
- `assembled_payload`

Row 1 shows the selected payload strand, selected complement strand, optional PWM motif layers, and mismatch annotations.

Row 2 shows the two inward-facing post-BsmBI split fragments built from the selected 4 nt junction window.

Row 3 shows those fragments reassembled back into the selected payload order.

The public contract is `split_yiu_payload_rendering_v4`.

### Command surface

```bash
uv run cruncher yiu init-workspace WORKSPACE
uv run cruncher yiu validate --spec configs/yiu/<workflow>.yiu.yaml
uv run cruncher yiu render --spec configs/yiu/<workflow>.yiu.yaml
uv run cruncher yiu render --spec configs/yiu/<workflow>.yiu.yaml --emit-renders
uv run cruncher yiu show --bundle outputs/<workflow>
```

`design` is not part of the public YIU surface.

### Bundle surface

YIU publishes one deterministic bundle under `output.bundle_dir`, typically `outputs/<workflow>/`.
`render` and `show` both consume one shared bundle-artifact surface and one shared bundle-state family so the CLI and app layer do not reconstruct bundle internals ad hoc.

The bundle contract is intentionally split across bundle truth, published view contracts, and composite render output. Use [YIU Artifacts](../reference/yiu_artifacts.md) for the exact emitted files, shared inspection fields, and render-status semantics.

### What `validate` checks

- the root contract and schema version match `split_yiu_payload_rendering_v4`
- exactly one input kind is populated
- the resolved payload sequence is present and contains valid IUPAC DNA characters
- the junction policy yields one valid internal 4 nt window with non-empty left and right payload bodies
- the mismatch policy is internally consistent and uses `strand_mode: per_position`
- PWM mode and PWM source are compatible with the input kind
- `sample_hit` provenance resolves to one exact payload sequence or fails fast
- PWM-aware optimization is deterministic and exhaustive across the allowed candidate space

### Visuals and inspection

The payload view uses `yiu_payload_visual_v1`.
The current YIU visual system is `bench_strip`: `payload` uses the `evidence_ribbon` direction, while `split_payload` and `assembled_payload` use `operator_strip`. Use [YIU Visual System](../reference/yiu_visual_system.md) for the hierarchy rationale and style-boundary rules.

When PWM context is available, the payload view includes motif layers aligned to payload-forward coordinates.
When PWM is absent or disabled, the same contract stays valid with an empty `motif_layers` list.

The three-view composite follows one explicit visual system:

- `payload` uses the `evidence_ribbon` direction so sequence truth, mismatch evidence, and motif overlays stay in one dense operator row
- `split_payload` and `assembled_payload` use the `operator_strip` direction so assembly geometry stays centered, legend-light, and subordinate to the payload truth row
- unknown view ids fail fast during style planning instead of silently inheriting a plausible strip preset

`sample_hit` source resolution follows the rules in [YIU Spec Reference](../reference/yiu_spec.md). Ambiguous or missing sources fail fast.

`cruncher yiu show` surfaces the bundle contract, provenance, selected payload and complement sequences, junction and mismatch summary, PWM mode, render status, integrity checks, and artifact paths. Use [YIU Artifacts](../reference/yiu_artifacts.md) for the exact inspection fields and fail-fast drift rules.

`show` is fail-fast on bundle drift: missing published view contracts, manifest/inventory disagreements, payload-view motif drift, a `rendered` bundle with a missing `payload_views.pdf`, or a configured published plot path that does not exist are treated as bundle corruption.

`cruncher yiu render --spec <workflow>.yiu.yaml --emit-renders` validates the spec, writes the payload bundle under `output.bundle_dir`, renders one composite `payload_views.pdf` page with the three canonical panels, mirrors that PDF to `output.published_plot_path` when configured, and updates `visual_inventory.json` in the same bundle directory.

The split middle row renders `split_payload_left` before `split_payload_right`. Each panel shows the retained post-digestion fragment, its inward-facing sticky end, selected-versus-canonical sticky-end metadata, the reverse-complemented payload-body slice, and optional ghosted excision context.

The assembled payload returns to original payload order. It publishes one explicit `junction_span` in payload coordinates and does not use a seam or ligation-boundary surrogate in the operator-facing contract.

### Maintainer boundaries

Keep this guide operator-first. The canonical module ownership map lives in [Architecture](../reference/architecture.md).

At the tool boundary, YIU publishes contracts and jobs; `baserender` consumes those contracts through its public API. Cross-tool integrations should not import `dnadesign.baserender.src.*`.

### Related docs

Start with [YIU Workspace Demo](../demos/demo_yiu_workspace.md), then use:

- [YIU Spec Reference](../reference/yiu_spec.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
- [CLI Reference](../reference/cli.md)
