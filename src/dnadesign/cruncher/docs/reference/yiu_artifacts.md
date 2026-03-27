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
- `yiu_trace_manifest.json`: stable trace summary with per-state mode and issue counts
- `yiu_published_views_manifest.json`: stable inventory of emitted neutral state contracts
- `yiu_parts.csv`: primary/complement sequence rows emitted per state
- `yiu_annotations.csv`: source-annotation table
- `yiu_fragments.csv`: nickase/cleanup fragment length table

`yiu_manifest.json` and `yiu_status.json` both publish reproducibility-facing metadata:

- `family: yiu`
- `protocol: yiu_v1` for legacy specs or the active `protocol_template` id for `schema_version: 2`
- `protocol_template` when the run used a `v2` protocol-template spec
- `state_count`
- `sequence_mode`
- `validation_mode`
- `engine_contract_version`
- `view_contract_version`
- `input_fingerprint`
- `catalog_fingerprint`
- `runtime_signature` with Cruncher/versioned contract provenance

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
Each view now declares `schema_version`, `family`, `protocol`, `state_kind`, `molecule_topology`, `sequence_mode`, and `validation_mode` before the state-local `meta` payload.
When `publish_contract_version: 2` is active, the view also declares `view_contract_version`, `protocol_template`, `topology_kind`, `segments`, `annotations`, `cuts`, `junctions`, and `fragments` while preserving the legacy `primary_sequence`, `complement_sequence`, and `meta` keys.
For PCR and restriction-digest states, the contract `meta` includes projected annotation coordinates for the current state. The digested-state contract also records explicit cut boundaries and removed-flank intervals.
For circularization, `meta.payload_junction_segments` and `meta.payload_junction` record how the authored source halves join into the assembled payload, while `paired_nt`, `unpaired_tail_nt`, `bulge_nt`, and aligned-core coordinates make the sticky-end compatibility call explicit.
For post-nickase and foldback states, `meta` includes retained-component mappings and projected homology-window coordinates relative to the retained-product state. Junction-spanning projections stay explicit through `parts[]` plus `spans_junction` rather than collapsing back to source coordinates.
For foldback specifically, `meta.paired_nt`, `meta.overlap_start`, `meta.overlap_end`, `meta.sequence_mode`, and `meta.topology_compatibility` separate topological compatibility from the raw reverse-complement overlap count.
For `y_adapter_ligated_product`, `meta` records `topology: branched_y`, ordered `arms`, and `branch_junction` so the Y-adapter geometry stays explicit in the published contract.
For the `v2` hairpin-template path, `hairpin_pcr_linear_insert.json` publishes the assembled payload as a compound projection across the retained-product junction and uses `topology_kind: linear_dsdna`.

### Status semantics

- `status: completed` means the explicit YIU validation report was satisfied
- `status: unsatisfied` means at least one structured issue was emitted, but the explicit bundle was still materialized for inspection
- `validation_mode: concrete_realization` means every validated sequence was concrete DNA
- `validation_mode: pattern_compatibility` means at least one validated state relied on IUPAC pattern compatibility rather than a single concrete realization
- `sequence_mode: pattern` in `v2` means at least one state remained a pattern over concrete realizations rather than one fixed molecule

See [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Spec Reference](yiu_spec.md) for schema details.
