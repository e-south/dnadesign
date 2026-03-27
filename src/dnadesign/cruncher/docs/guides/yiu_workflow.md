## YIU Workflow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-26

The YIU family models a protocol graph of molecular state transitions rather than a single construct. The explicit tracer-bullet lane validates a strict `.yiu.yaml` spec, walks the intended state graph, and materializes a deterministic bundle under `outputs/yiu/explicit/`.

Two explicit contracts now coexist:

- `schema_version: 1` with `protocol: yiu_v1` preserves the shipped compatibility lane.
- `schema_version: 2` with `protocol_template: msd_hop_retron_eco1_v1` models the corrected msd-HOP/Y-adapter hairpin workflow with typed intermediates and additive published view contract `2`.

### Command surface

```bash
uv run cruncher yiu init-workspace WORKSPACE
uv run cruncher yiu validate --spec configs/yiu/example.yiu.yaml
uv run cruncher yiu design --spec configs/yiu/example.yiu.yaml
uv run cruncher yiu trace --spec configs/yiu/example.yiu.yaml
uv run cruncher yiu show --run outputs/yiu/explicit/<spec.name>/<design_id>
```

### Modeled state graph

The legacy `v1` lane publishes one ordered state graph:

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

The template-correct `v2` lane publishes:

1. `source_oligo_ssdna`
2. `source_amplicon_dsdna`
3. `post_double_nicking_fragment_pool`
4. `post_heat_cleanup_fragment_pool`
5. `adapter_annealed_complex`
6. `ligated_ssdna_hairpin`
7. `hairpin_pcr_linear_insert`
8. `post_insert_cleanup_linear_insert` when enabled
9. `backbone_amplicon` when `workflow_scope: insert_plus_backbone_cloning`
10. `assembly_reaction` when Golden Gate assembly is enabled
11. `assembled_plasmid_candidate` when cloning completes

Cruncher validates protocol compatibility, not stochastic wet-lab yield. Each state records the intended primary sequence or retained-product view plus step-local metadata such as overhangs, fragment lengths, or foldback homology.
For transformed intermediates, the explicit lane now publishes cut-retained state views rather than leaving every downstream step in source coordinates: PCR records amplicon-projected annotations, restriction digest records the retained duplex interval, cut boundaries, removed flanks, and digested-state annotation projections, circularization records payload-junction segments plus explicit sticky-end geometry (`paired_nt`, `unpaired_tail_nt`, `bulge_nt`), nickase fragmentation records retained-component mappings that subsequent foldback validation uses, foldback records `paired_nt`, `overlap_start`, `overlap_end`, `sequence_mode`, and `topology_compatibility`, and adapter ligation records explicit branched-arm geometry for the Y product.
The `v2` path adds typed topology/state fields and compound region projections, so payload assembly across a junction is published explicitly in assembled coordinate space rather than flattened back to one source segment.
Every report and published state view also declares `sequence_mode` plus `validation_mode`: concrete inputs stay `concrete_realization`, while active IUPAC ambiguity upgrades the explicit lane to `pattern_compatibility`. In `v2`, pattern-mode states also publish `pattern_evidence_summary`.

### What `validate` checks

- source annotations resolve onto the source oligo unambiguously
- retained and sacrificial regions must not overlap; partially overlapping retained-only or sacrificial-only regions are rejected before the state graph runs
- `v2` also rejects payload-vs-sacrificial overlap and primer-binding-core overlap with nickase/restriction sites
- the PCR state records the primer-bounded amplicon and flags authored downstream annotations that fall outside that amplicon
- restriction sites match the requested sequence/orientation, produce the expected sticky ends, and materialize the cut-retained duplex with explicit removed-flank metadata
- circularization compatibility uses deterministic sticky-end geometry, not thermodynamic scoring
  `exact_complement` requires full reverse-complement identity with no slack.
  `partial_complement` requires one contiguous reverse-complement core that satisfies `min_paired_nt` and `max_unpaired_tail_nt`.
  `bulged` allows one bounded internal insertion/deletion in addition to the partial-complement rules.
- referenced catalog entries exist and agree with the authored restriction sites, nickase sites, and adapter settings
- the `assembled_payload` derived from `payload_goal.left_half_ref` and `payload_goal.right_half_ref` matches the goal sequence
- the circularization state records the authored left/right payload halves as explicit `payload_junction_segments` plus the join index implied by `payload_goal.junction_rule`
- nickase sites do not cut retained regions
- each configured sacrificial region is actually cut by the referenced nickase sites; otherwise the explicit lane emits `NICKASE_SACRIFICIAL_REGION_UNCUT`
- the retained-product state records each retained component with source and retained-product coordinates so downstream steps can inspect the explicit mapping
- sacrificial fragment lengths satisfy the configured cleanup assumptions, including `min_removed_fragment_nt` and `max_retained_sacrificial_fragment_nt`
- adapter ligation can resolve its sequence from the step, from `adapter_policy.adapter_sequence`, or from `adapter_policy.y_adapter_id` plus `catalogs.adapters`, and the resulting `branched_y` state publishes arm coordinates plus `branch_junction`
- foldback windows expose the configured reverse-complement homology on the retained-product state; windows that were discarded by upstream transforms emit `HOMOLOGY_WINDOW_EXCLUDED_FROM_CURRENT_STATE`, while windows that project across the retained-product junction stay explicit via `parts[]` / `spans_junction` and fail fast with `HOMOLOGY_WINDOW_SPANS_JUNCTION`
- the downstream amplifiable product preserves the assembled payload and contains the required primer-binding motifs
- `v2` pattern checks resolve to `guaranteed`, `possible`, or `impossible`; `require_guaranteed` fails on merely possible matches, while `allow_possible_with_warning` keeps the run satisfied but warns
- `v2` `adapter_anneal` and `hairpin_ligation` use the same exact/partial/bulged compatibility model as the old circularization check, but on typed annealed/hairpin states

### Bundle shape

`design` and `trace` both materialize the explicit state bundle. The key difference is operational intent:

- `design` is the default explicit artifact command
- `trace` emphasizes state-by-state inspection of the modeled protocol graph

Both commands publish per-state neutral contracts in `published/views/` and summary tables at the run root. The state graph is the contract boundary; renderer coupling stays file-based.
They also write `yiu_trace_manifest.json` and `yiu_published_views_manifest.json` so downstream QA can inventory states, modes, and view paths without replaying the full JSONL trace.
For `publish_contract_version: 2`, each published state also carries `view_contract_version`, `state_kind`, `topology_kind`, `segments`, `annotations`, `cuts`, `junctions`, and `fragments` in addition to the legacy `primary_sequence`, `complement_sequence`, and `meta`.

Start with the scaffolded walk-through in [YIU Workspace Demo](../demos/demo_yiu_workspace.md), then use the strict references below:

- [YIU Spec Reference](../reference/yiu_spec.md)
- [YIU Artifacts](../reference/yiu_artifacts.md)
