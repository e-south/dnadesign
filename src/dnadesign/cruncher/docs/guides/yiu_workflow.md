## YIU Workflow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-26

The YIU family models a protocol graph of molecular state transitions rather than a single construct. The explicit tracer-bullet lane validates a strict `.yiu.yaml` spec, walks the intended state graph, and materializes a deterministic bundle under `outputs/yiu/explicit/`.

### Command surface

```bash
uv run cruncher yiu init-workspace WORKSPACE
uv run cruncher yiu validate --spec configs/yiu/example.yiu.yaml
uv run cruncher yiu design --spec configs/yiu/example.yiu.yaml
uv run cruncher yiu trace --spec configs/yiu/example.yiu.yaml
uv run cruncher yiu show --run outputs/yiu/explicit/<spec.name>/<design_id>
```

### Modeled state graph

The phase-1 explicit lane publishes one ordered state graph:

1. `source_oligo_ssdna`
2. `pcr_linear_duplex`
3. `digested_linear_duplex`
4. `circularization_candidate`
5. `post_exonuclease_enriched_pool`
6. `post_nickase_fragmentation`
7. `post_size_selection`
8. `foldback_or_cap_intermediate`
9. `y_adapter_ligated_product`
10. `downstream_amplifiable_product`

Cruncher validates protocol compatibility, not stochastic wet-lab yield. Each state records the intended primary sequence or retained-product view plus step-local metadata such as overhangs, fragment lengths, or foldback homology.

### What `validate` checks

- source annotations resolve onto the source oligo unambiguously
- restriction sites match the requested sequence/orientation and produce the expected sticky ends
- the `assembled_payload` derived from `payload_goal.left_half_ref` and `payload_goal.right_half_ref` matches the goal sequence
- nickase sites do not cut retained regions
- sacrificial fragment lengths satisfy the configured cleanup assumptions
- foldback windows expose the configured reverse-complement homology
- the downstream amplifiable product contains the required primer-binding motifs

### Bundle shape

`design` and `trace` both materialize the explicit state bundle. The key difference is operational intent:

- `design` is the default explicit artifact command
- `trace` emphasizes state-by-state inspection of the modeled protocol graph

Both commands publish per-state neutral contracts in `published/views/` and summary tables at the run root. The state graph is the contract boundary; renderer coupling stays file-based.

Start with the scaffolded walk-through in [YIU Workspace Demo](../demos/demo_yiu_workspace.md), then use the strict references below:

- [YIU Spec Reference](../reference/yiu_spec.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
