## YIU Artifacts

**Audience:** YIU workflow users and maintainers
**Applies to:** `uv run cruncher yiu render|show`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-04

YIU writes one payload-centric bundle family under the workspace-relative `output.bundle_dir` path.
This page is the source of truth for emitted files, render-status semantics, and the shared `render`/`show` inspection surface.

### Bundle truth vs mirror

- `output.bundle_dir` is the bundle-local source of truth for YIU artifacts.
- `output.published_plot_path` is an optional workspace-level mirror of the bundle PDF.
- The CLI and app layer should read bundle truth from `visual_inventory.json` and `bundle_manifest.json`, not from filesystem guesswork.

Recommended patterns:

- YIU-only workspace: `outputs/<workflow>/` plus `output.published_plot_path: outputs/<workflow>__payload_views.pdf`
- Sample-backed workspace: `outputs/plots/yiu__<workflow>/` with no second workspace-level mirror

Each bundle uses `visual_inventory.json` as the operator-facing render-state record.

It records:

- `split_yiu_payload_bundle_v4`
- view contract paths
- explicit `visual_direction` per published view
- render artifact paths
- bundle composite render artifact path
- published plot artifact path when configured
- renderer kind
- view ids
- render request and completion truth
- `render_count`
- `render_status`
- `last_rendered_at`

### Bundle layout

```text
outputs/<workflow>/
  bundle_manifest.json
  normalized_payload.json
  visual_inventory.json
  payload_view.json
  split_payload_view.json
  assembled_payload_view.json
  payload_views.pdf
  baserender_jobs/
    *.job.yaml     # only when emit_render_jobs_debug: true
```

`bundle_manifest.json` uses the `split_yiu_payload_bundle_v4` contract and records:

- input contract and input kind
- payload label when available
- payload length
- selected payload and complement sequences
- junction summary
- mismatch plan
- PWM mode and whether PWM scoring was effective
- provenance
- published view entries
- one bundle-level `composite_render_artifact_path`
- one optional workspace-level `published_plot_artifact_path`
- render status
- one operator-facing composite PDF under `payload_views.pdf`

`normalized_payload.json` is the normalized internal object serialized for inspection and downstream validation.

Published contract paths:

- `payload_view.json`
- `split_payload_view.json`
- `assembled_payload_view.json`

The split and assembled views stay sequence-centric; the payload view uses `yiu_payload_visual_v1` and is the only place motif layers appear.

The payload visual contract carries:

- reference payload row visibility
- selected payload and complement rows
- junction annotations
- mismatch annotations
- optional motif layers aligned to payload-forward coordinates

### Shared bundle surface

`cruncher yiu render` and `cruncher yiu show` share one bundle-artifact surface:

- `bundle_dir`
- `outputs_root`
- `composite_render_artifact_path`
- `published_plot_artifact_path`
- `bundle_manifest_path`
- `normalized_payload_path`
- `visual_inventory_path`

That shared surface is intentional: bundle layout changes should land once in the app layer, not be reconstructed independently in CLI commands. The workflow guide points here instead of duplicating the inspection-field list.
Each view entry also records one explicit `visual_direction` so downstream tools do not infer layout policy from `view_id` or showcase defaults.

### Status semantics

- `render_status: not_requested` means the bundle was published without PDF rendering
- `render_status: rendered` means all three payload views rendered successfully
- `render_status: failed` means BaseRender failed and no substitute renders were fabricated
- `cruncher yiu show` rejects bundles whose manifest, normalized payload, inventory, or published artifact paths disagree

### Operator inspection

`cruncher yiu show` surfaces:

- bundle directory and bundle contract
- provenance
- selected payload length
- selected junction and mismatch plan
- PWM mode and effective status
- render summary from `visual_inventory.json`
- composite render path when available
- published plot path when configured
- key artifact paths

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance, [YIU Visual System](yiu_visual_system.md) for named visual directions and hierarchy, and [YIU Spec Reference](yiu_spec.md) for schema details.
