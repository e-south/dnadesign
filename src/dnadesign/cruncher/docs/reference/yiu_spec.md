## YIU Spec Reference

**Audience:** YIU workflow users and maintainers
**Applies to:** `configs/yiu/*.yiu.yaml`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-26

YIU specs are strict YAML documents rooted at `yiu:`. The phase-1 explicit lane uses schema version `1` and protocol `yiu_v1`.

### Minimal layout

```yaml
yiu:
  schema_version: 1
  protocol: yiu_v1
  name: demo_yiu
  source_oligo:
    sequence: AAAA...
    primer_sites: [...]
    restriction_sites: [...]
    nickase_sites: [...]
    payload_windows: [...]
    homology_windows: [...]
    retained_regions: [...]
    sacrificial_regions: [...]
  step_graph:
    steps: [...]
  payload_goal:
    assembled_payload: TTAACCGG
    left_half_ref: left_half
    right_half_ref: right_half
    junction_rule: contiguous_after_ligation
  cleanup_policy: {...}
  adapter_policy: {...}
  catalogs: {...}
  output:
    run_dir: outputs/yiu/explicit
```

### Required concepts

- `source_oligo.sequence`: concrete or IUPAC DNA sequence for the authored source oligo
- `step_graph`: fixed phase-1 graph from `pcr` through `amplification`
- `assembled_payload`: the target retained payload after ligation/junction assembly
- `payload_goal.left_half_ref` / `payload_goal.right_half_ref`: named payload windows on the source oligo
- `catalogs`: optional workspace-relative protocol catalogs for restriction enzymes, nickases, and adapters
- `adapter_ligation`: the adapter sequence may come from `step.adapter_sequence`, `adapter_policy.adapter_sequence`, or `adapter_policy.y_adapter_id` plus `catalogs.adapters`
- `cleanup_policy.size_selection`: thresholds used to validate sacrificial fragment removal and retained-product survival
  `min_removed_fragment_nt` is the smallest sacrificial fragment length the modeled cleanup step assumes can be removed.
  `max_retained_sacrificial_fragment_nt` is the largest sacrificial fragment length still compatible with the cleanup assumption.
  `min_retained_product_nt` is the minimum retained-product length expected to survive cleanup.

### Catalog file shapes

If you set a `catalogs.*` path, Cruncher validates both the file schema and the referenced entries.

Restriction enzyme catalog:

```yaml
restriction_enzymes:
  entries:
    - id: BsaI
      recognition_sequence: GGTCTC
      top_cut_offset: 6
      bottom_cut_offset: 10
```

Nickase catalog:

```yaml
nickases:
  entries:
    - id: Nt.Mock
      recognition_sequence: GGGG
      top_cut_offset: 2
```

Adapter catalog:

```yaml
adapters:
  entries:
    - id: demo_y_adapter
      sequence: AGATCGGA
```

Catalog offsets are optional. When present, they must agree with the site geometry authored in the YIU spec.

### Important invariants

- annotation ids must be unique across primer sites, enzyme sites, payload windows, homology windows, retained regions, and sacrificial regions
- the PCR step records `amplicon_start`, `amplicon_end`, and `amplicon_length_nt`, and emits a structured issue if downstream annotations fall outside that primer-defined amplicon
- restriction and nickase sites must match the source oligo in the requested orientation
- referenced catalog entries must exist and match the authored enzyme motifs, cut offsets, and adapter sequences
- `step_graph.steps` must use the fixed YIU order for the explicit tracer bullet
- `output.run_dir` must stay inside the workspace root

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Artifacts](yiu_artifacts.md) for the emitted bundle contract.
