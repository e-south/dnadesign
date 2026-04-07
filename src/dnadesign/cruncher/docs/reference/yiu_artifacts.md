## YIU Artifacts

**Audience:** YIU workflow users and maintainers
**Applies to:** `uv run cruncher yiu render|show`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-07
**Last updated by:** cruncher-maintainers on 2026-04-07

A successful YIU render writes one deterministic bundle under the workspace-relative `output.bundle_dir` path. Treat that bundle directory as the source of truth. `output.published_plot_path` is only an optional mirror of the composite PDF. Use this reference for emitted files, render-status semantics, and the shared `render`/`show --verbose` inspection surface.

Use the workflow guide for command flow and solver behavior. Use the spec reference for `.yiu.yaml` rules. This page is only about the published bundle and the checks around it.

<!-- docs:toc:off -->

Use other pages for:

- [YIU Workflow](../guides/yiu_workflow.md)
- [YIU Spec Reference](yiu_spec.md)
- [YIU Visual System](yiu_visual_system.md)

### Bundle truth vs mirror

- `output.bundle_dir` is the bundle-local source of truth for YIU artifacts.
- `output.published_plot_path` is an optional workspace-level mirror of the bundle PDF.
- The CLI and app layer should read bundle truth from `visual_inventory.json` and `bundle_manifest.json`, not from filesystem guesswork.

Recommended patterns:

- YIU-only workspace: `outputs/<workflow>/` plus `output.published_plot_path: outputs/<workflow>__payload_views.pdf`
- Sample-backed workspace: `outputs/plots/yiu__<workflow>/` with no second workspace-level mirror

Each bundle uses `visual_inventory.json` as the machine-facing render-state and integrity record.

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
  bundle_summary.json
  bundle_manifest.json
  normalized_payload.json
  visual_inventory.json
  payload_view.json
  split_payload_view.jsonl   # JSONL rows: split_payload_left then split_payload_right
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

`bundle_summary.json` is the concise handoff surface for one run. It keeps the main sequence story in one place:

- one `views` block for `payload`, `split_left`, `split_right`, and `assembled`
- reference and mismatch-present variants for each view
- one reference duplex and one mismatch-present duplex view for every published handoff row
- top and bottom strand sequences in explicit 5' to 3' orientation for every view
- one `changed_rows` list per view so the mismatch-bearing row is obvious at a glance
- one `overhang_5to3` block with reference and mismatch-present sticky ends
- one `junction_payload_sequence_5to3` block for the 4 bp payload window
- compact strand-aware mismatch notation using 1-based payload positions
- raw mismatch list, PWM summary, published view ids, and render status

Inside `sequence_summary`, the operator handoff is published as:

- `views.payload`
- `views.split_left`
- `views.split_right`
- `views.assembled`

Each view entry carries:

- `canonical.top_strand_5to3`
- `canonical.bottom_strand_5to3`
- `mismatch_present.top_strand_5to3`
- `mismatch_present.bottom_strand_5to3`
- `changed_rows`

`normalized_payload.json` is the normalized semantic payload object serialized for inspection and downstream validation. It does not mirror bundle file layout or render artifacts.

`bundle_manifest.json` and `visual_inventory.json` are machine-facing ledgers. They exist so `render`, `show`, and downstream integrity checks can agree on published contracts, render status, and artifact paths without guessing from the filesystem.

Published contract paths:

- `payload_view.json`
- `split_payload_view.jsonl` (JSONL)
- `assembled_payload_view.json`

The split and assembled views stay sequence-centric machine contracts; the payload view uses `yiu_payload_visual_v1` and is the only place motif layers appear.

The payload visual contract carries:

- reference payload row visibility
- selected payload and complement rows
- junction annotations
- mismatch annotations
- optional motif layers aligned to payload-forward coordinates

### Shared bundle surface

`cruncher yiu render` and `cruncher yiu show --verbose` share one machine-facing bundle-artifact surface:

- `bundle_dir`
- `outputs_root`
- `composite_render_artifact_path`
- `published_plot_artifact_path`
- `bundle_summary_path`
- `bundle_manifest_path`
- `normalized_payload_path`
- `visual_inventory_path`

That shared surface is intentional: bundle layout changes should land once in the app layer, not be reconstructed independently in CLI commands. The workflow guide points here instead of restating the inspection-field list.

Each view entry also records one explicit `visual_direction` so downstream tools do not infer layout policy from `view_id` or showcase defaults.

### Status semantics

- `render_status: not_requested` means the bundle was published without PDF rendering
- `render_status: rendered` means all three payload views rendered successfully
- `render_status: missing` means the bundle expects rendered outputs but they are absent on disk
- `render_status: partial` means at least one render completed but the full bundle render set is incomplete
- `render_status: failed` means BaseRender failed and no substitute renders were fabricated
- `cruncher yiu show` rejects bundles whose manifest, normalized payload, inventory, or published artifact paths disagree

### Operator inspection

`cruncher yiu show` surfaces:

- bundle directory
- selected payload length
- one concise sequence summary with payload, split-left, split-right, and assembled views in explicit 5' to 3' top/bottom rows
- one explicit reference-vs-mismatch-present overhang summary
- compact mismatch edits using `PS` for the displayed payload strand and `AS` for the opposite strand
- one ligation summary line with profile, whether ligation-aware ranking applied, chosen mismatch classes, position classes, and the decision note
- selected junction and mismatch plan
- PWM mode and effective status
- no machine-facing bundle ledger paths by default

Default `show --json` keeps the operator bundle surface and omits machine ledger paths, normalized payload detail, `motif_context`, `optimization_decision`, and `split_row_debug` unless `--verbose` is set.
Human-readable `show --verbose` adds provenance, bundle contract, render and integrity detail, machine-facing artifact paths, and split-row debug lines; the optimizer trace and PWM context remain JSON-only.

### Integrity checks

`show` treats the bundle as corrupt when:

- the manifest and inventory disagree
- a published view contract is missing
- `normalized_payload.json` and the published views disagree on sequence or mismatch truth
- PWM motif layers drift away from the normalized payload
- `render_status: rendered` is recorded but `payload_views.pdf` or a configured published plot is missing

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance, [YIU Visual System](yiu_visual_system.md) for named visual directions and hierarchy, and [YIU Spec Reference](yiu_spec.md) for schema details.
