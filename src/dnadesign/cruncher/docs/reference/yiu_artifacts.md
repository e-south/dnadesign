## YIU Artifacts

**Audience:** YIU workflow users and maintainers
**Applies to:** `uv run cruncher yiu render|show`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-03

YIU writes one payload-centric bundle family under `bundles/<workflow>/`.

Each bundle uses `visual_inventory.json` to track render status and published artifact paths.

It records:

- `split_yiu_payload_bundle_v4`
- view contract paths
- render artifact paths
- bundle composite render artifact path
- renderer kind
- view ids
- render request and completion truth
- `render_count`
- `render_status`
- `last_rendered_at`

### Bundle layout

```text
bundles/<workflow>/
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
- render status
- one operator-facing composite PDF under `payload_views.pdf`

`normalized_payload.json` is the canonical internal object serialized for inspection and downstream validation.

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
- key artifact paths

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Spec Reference](yiu_spec.md) for schema details.
