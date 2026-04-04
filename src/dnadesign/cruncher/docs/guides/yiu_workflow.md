## YIU Workflow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-03

YIU is a payload-centric rendering workflow with a strict v4 contract.

The public lane is:

`input payload -> normalized payload -> optimized junction/mismatch plan -> canonical bundle -> BaseRender`

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

When PWM context is available, the payload view includes motif layers aligned to payload-forward coordinates.
When PWM is absent or disabled, the same contract stays valid with an empty `motif_layers` list.

`sample_hit` can source payloads from:

- direct `payload_sequence`
- workspace-local `source_artifact_path`
- sibling workspace public artifacts through `metadata.source_workspace` + `metadata.source_artifact`

Relative `source_artifact_path` traversal stays inside the current workspace; sibling workspaces are resolved only through the explicit metadata pair above.

`cruncher yiu show` surfaces:

- bundle directory and bundle contract
- provenance
- selected payload and complement sequences
- selected junction and mismatch summary
- PWM mode, effective status, and fallback reason when applicable
- render status from `visual_inventory.json`
- integrity checks against the manifest, inventory, normalized payload, and published view contracts
- composite render path when present
- optional verbose split-row debug details when `--verbose` is requested

`show` is fail-fast on bundle drift: missing published view contracts, manifest/inventory disagreements, payload-view motif drift, a `rendered` bundle with a missing `payload_views.pdf`, or a configured published plot path that does not exist are treated as bundle corruption.

`cruncher yiu render --spec <workflow>.yiu.yaml --emit-renders` validates the spec, writes the payload bundle under `output.bundle_dir`, renders one composite `payload_views.pdf` page with the three canonical panels, mirrors that PDF to `output.published_plot_path` when configured, and updates `visual_inventory.json` in the same bundle directory.

The split middle row renders `split_payload_left` before `split_payload_right`. Each panel shows the retained post-digestion fragment, its inward-facing sticky end, selected-versus-canonical sticky-end metadata, the reverse-complemented payload-body slice, and optional ghosted excision context.

The assembled payload returns to original payload order. It publishes one explicit `junction_span` in payload coordinates and does not use a seam or ligation-boundary surrogate in the operator-facing contract.

Start with [YIU Workspace Demo](../demos/demo_yiu_workspace.md), then use:

- [YIU Spec Reference](../reference/yiu_spec.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
