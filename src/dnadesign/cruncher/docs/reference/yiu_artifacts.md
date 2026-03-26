## YIU Artifacts

**Audience:** YIU workflow users and maintainers
**Applies to:** `uv run cruncher yiu design|trace`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-26

Explicit YIU runs are written under:

```text
<workspace>/outputs/yiu/explicit/<spec.name>/<design_id>/
```

### Core files

- `yiu_report.json`: structured validation and state-trace report
- `yiu_status.json`: lightweight status summary for operators
- `yiu_manifest.json`: run-level manifest with artifact inventory
- `yiu_trace.jsonl`: one JSON record per modeled state
- `yiu_parts.csv`: primary/complement sequence rows emitted per state
- `yiu_annotations.csv`: source-annotation table
- `yiu_fragments.csv`: nickase/cleanup fragment length table

### Published state contracts

When `output.emit_view_contracts: true`, Cruncher writes per-state neutral contracts under:

```text
published/views/
  source_oligo_ssdna.json
  pcr_linear_duplex.json
  digested_linear_duplex.json
  circularization_candidate.json
  post_exonuclease_enriched_pool.json
  post_nickase_fragmentation.json
  post_size_selection.json
  foldback_or_cap_intermediate.json
  y_adapter_ligated_product.json
  downstream_amplifiable_product.json
```

These are workflow-owned JSON contracts, not renderer-internal payloads.

### Status semantics

- `status: completed` means the explicit YIU validation report was satisfied
- `status: unsatisfied` means at least one structured issue was emitted, but the explicit bundle was still materialized for inspection

See [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Spec Reference](yiu_spec.md) for schema details.
