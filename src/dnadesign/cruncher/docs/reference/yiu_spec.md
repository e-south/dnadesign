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
- `cleanup_policy.size_selection`: thresholds used to validate sacrificial fragment removal and retained-product survival

### Important invariants

- annotation ids must be unique across primer sites, enzyme sites, payload windows, homology windows, retained regions, and sacrificial regions
- restriction and nickase sites must match the source oligo in the requested orientation
- `step_graph.steps` must use the fixed YIU order for the explicit tracer bullet
- `output.run_dir` must stay inside the workspace root

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Artifacts](yiu_artifacts.md) for the emitted bundle contract.
