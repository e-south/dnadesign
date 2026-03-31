## YIU Workflow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-29

YIU is a replay/validation engine with bounded scaffold solving.

It ships one v4 workflow lane:

- `schema_version: 4`
- `protocol_template: yiu_circularized_payload_v1`
- explicit materialization through `cruncher yiu trace`
- bounded solve through `cruncher yiu solve`
- bundle-local `visual_inventory.json` as the render source of truth

### Command surface

```bash
uv run cruncher yiu init-workspace WORKSPACE
uv run cruncher yiu validate --spec configs/yiu/<workflow>.yiu.yaml
uv run cruncher yiu trace --spec configs/yiu/<workflow>.yiu.yaml
uv run cruncher yiu trace --spec configs/yiu/<workflow>.yiu.yaml --emit-renders
uv run cruncher yiu solve --spec configs/yiu/<workflow>.yiu.solve.yaml
uv run cruncher yiu show --run outputs/yiu/explicit/<workflow>/<trace_id>
uv run cruncher yiu show --run outputs/yiu/solve/<workflow>/<solve_id>
uv run cruncher yiu render --run outputs/yiu/explicit/<workflow>/<trace_id>
uv run cruncher yiu render --run outputs/yiu/solve/<workflow>/<solve_id>
```

`design` is not part of the public YIU surface.

### State graph

The `yiu_circularized_payload_v1` workflow emits 9 states:

1. `source_oligo_ssdna`
2. `pcr_linear_duplex`
3. `type_iis_cut_product_duplex`
4. `circularized_payload_candidate`
5. `post_sacrificial_fragmentation`
6. `post_fragment_cleanup`
7. `snapback_adapter_complex`
8. `ligated_ssdna_hairpin`
9. `hairpin_pcr_linear_insert`

`post_exonuclease_cleanup` is method metadata only. It is not an emitted state in v4.

### What `validate` checks

- the source primary strand follows the fixed owner order for `yiu_circularized_payload_v1`
- every emitted nucleotide has exactly one structural owner on each emitted strand
- effect-tag kinds are closed and unknown operational tags fail at load time
- owner/tag overlap legality fails closed
- `type_iis_cut_product_duplex` is a real cut-product state with sticky-end metadata
- payload assembly remains interpretable where the payload exists
- the `Nt.Bpu10I` local context, nick boundary, and exposed tether geometry all pass
- sacrificial fragmentation stays within the declared fragment limit

### Solve lane

`cruncher yiu solve` resolves one base spec, derives payload halves and payload overhangs from a single target payload, and mutates only:

- the payload target or pattern
- the payload bulge mask
- declared windows inside `sacrificial_region_long`

It does not mutate fixed scaffold owners, enzyme identities, tether or snapback owners, or the Y adapter sequence.

Default solve behavior is SAT-first:

- search all bounded candidates
- return `solved`, `unsatisfied`, or `incomplete_search`
- materialize one deterministic solution by default
- keep comparison solutions behind `compare_solutions: true`

The checked-in demo workspace is exhaustive under its current search bounds. On 2026-03-29 it found 2 satisfying solutions and selected 1 deterministic solution.

### Visuals and inspection

Every emitted state publishes `sequence_evidence_map_v1` and renders through BaseRender's `nucleotide_evidence_map` surface.

`cruncher yiu show` surfaces:

- bundle kind
- protocol template
- schema version or solve status
- final-state or solve summary
- exhaustive-search truth for solve runs
- hard-invariant summary for the selected final state
- render summary from `visual_inventory.json`
- key artifact paths

`cruncher yiu render --run <bundle>` rereads `visual_inventory.json`, regenerates missing PDFs directly into the bundle's published `visuals/` paths, and updates render truth in the same inventory.

Start with [YIU Workspace Demo](../demos/demo_yiu_workspace.md), then use:

- [YIU Spec Reference](../reference/yiu_spec.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
