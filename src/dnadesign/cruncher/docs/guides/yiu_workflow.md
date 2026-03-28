## YIU Workflow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-27

YIU is now a real workflow family with two explicit lanes and one bounded solve lane.

- `schema_version: 1` with `protocol: yiu_v1` remains the compatibility lane.
- `schema_version: 2` with `protocol_template: yiu_adapter_hairpin_v1` is the typed adapter-hairpin compatibility lane.
- `schema_version: 2` with `protocol_template: yiu_circularized_payload_v1` is the recommended canonical lane.
- `.yiu.solve.yaml` drives bounded search over declared variable windows and admits hits only after the explicit validator succeeds.

Ship posture for this tranche:

> YIU currently ships as a canonical circularized workflow family with bounded solve over declared source windows. Compatibility templates remain supported, but the canonical circularized template is the recommended operator path.

### Command surface

```bash
uv run cruncher yiu init-workspace WORKSPACE
uv run cruncher yiu validate --spec configs/yiu/example_canonical_circularized.yiu.yaml
uv run cruncher yiu design --spec configs/yiu/example_canonical_circularized.yiu.yaml
uv run cruncher yiu trace --spec configs/yiu/example_canonical_circularized.yiu.yaml
uv run cruncher yiu show --run outputs/yiu/explicit/example_canonical_circularized/<design_id>
uv run cruncher yiu solve --spec configs/yiu/example_canonical_circularized.yiu.solve.yaml
```

`design` and `trace` are currently operational aliases. They both materialize the same explicit bundle, but operators use `design` as the default artifact command and `trace` when they want state-graph inspection intent to be explicit.

### Recommended explicit state graph

The canonical circularized `v2` lane publishes these states:

1. `source_oligo_ssdna`
2. `pcr_linear_duplex`
3. `type_iis_digest_linear_duplex`
4. `circularized_payload_candidate`
5. `post_exonuclease_cleanup`
6. `post_sacrificial_fragmentation`
7. `post_fragment_cleanup`
8. `snapback_adapter_complex`
9. `ligated_ssdna_hairpin`
10. `hairpin_pcr_linear_insert`

The adapter-hairpin compatibility lane still publishes `source_oligo_ssdna`, `source_amplicon_dsdna`, `post_double_nicking_fragment_pool`, `post_heat_cleanup_fragment_pool`, `adapter_annealed_complex`, `ligated_ssdna_hairpin`, and `hairpin_pcr_linear_insert`, with optional insert-cleanup and cloning states when enabled.

### What `validate` checks

- source annotations resolve onto the authored source oligo
- overlap legality is enforced before state materialization
- canonical circularized specs must declare `template_bindings`; runtime no longer depends on hidden favorite IDs
- the explicit validator records `sequence_mode`, `validation_mode`, and per-state `pattern_evidence_summary`
- hard invariants are no longer silently accepted; supported checks are evaluated and unsupported scope requests fail fast
- split-template payload assembly is checked in `circularized_payload_junction`
- split-template publication uses `publish_contract_version: 3` by default and emits render-oriented contracts plus file-based BaseRender jobs when requested
- `output.emit_baserender_jobs: true` requires `output.emit_view_contracts: true`

### Solve lane

`cruncher yiu solve` reads `<workspace>/configs/yiu/<name>.yiu.solve.yaml`, resolves `base_spec`, enumerates only the declared variable windows, and runs every candidate through the full explicit YIU validator.

A candidate becomes a hit only if:

1. every declared window is concrete,
2. the explicit validator succeeds,
3. every declared hard invariant is `guaranteed`,
4. no unsupported invariant class is required,
5. and no hard explicit-lane issues remain.

Solve artifacts are written under `outputs/yiu/solve/<solve_name>/<solve_id>/`. Each materialized hit under `hits/hit_0001/` is a standard explicit YIU bundle, so `cruncher yiu show` can inspect both solve bundles and per-hit explicit bundles without a different code path.

### Operator surfaces

`cruncher yiu show` now surfaces:

- bundle kind (`explicit` or `solve`)
- run id / solve id
- step/state/issue counts for explicit bundles
- accepted/materialized hit counts plus final-state kind for solve bundles
- `published/views/`
- `published/visual_manifest.json`
- `published/baserender_jobs/`
- `published/renders/`
- solve-level `accepted_hits.jsonl` and the first materialized hit path when the run is a solve bundle
- top-hit explicit bundle roots for solve bundles

Use `cruncher yiu show --json` when you want the full normalized artifact inventory instead of the human summary.

Start with the scaffolded walk-through in [YIU Workspace Demo](../demos/demo_yiu_workspace.md), then use the references below:

- [YIU Spec Reference](../reference/yiu_spec.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
