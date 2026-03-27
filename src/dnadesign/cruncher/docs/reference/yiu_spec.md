## YIU Spec Reference

**Audience:** YIU workflow users and maintainers
**Applies to:** `configs/yiu/*.yiu.yaml`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-26

YIU specs are strict YAML documents rooted at `yiu:`. Cruncher now supports two explicit contracts:

- `schema_version: 1` with `protocol: yiu_v1` remains the shipped legacy compatibility lane.
- `schema_version: 2` with `family: yiu` and `protocol_template: msd_hop_retron_eco1_v1` is the corrected protocol-template path.

### Minimal layout

Legacy `v1`:

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

Template-correct `v2`:

```yaml
yiu:
  schema_version: 2
  family: yiu
  protocol_template: msd_hop_retron_eco1_v1
  workflow_scope: core_insert_generation
  name: demo_yiu_v2
  source_oligo:
    sequence: AAAA...
    annotations:
      primer_binding_cores: [...]
      primer_tails: [...]
      nickase_sites: [...]
      payload_windows: [...]
      homology_windows: [...]
      retained_regions: [...]
      sacrificial_regions: [...]
  steps:
    source_pcr: {...}
    double_nicking_digest: {...}
    heat_cleanup: {...}
    adapter_anneal: {...}
    hairpin_ligation: {...}
    hairpin_pcr: {...}
    insert_cleanup: {...}
    backbone_pcr: {...}
    golden_gate_assembly: {...}
  payload_goal:
    assembled_payload_pattern: TTAACCGG
    left_half_ref: payload_left
    right_half_ref: payload_right
    assembly_space: post_ligation
    evidence_policy: require_guaranteed
  catalogs:
    enzymes: catalogs/enzymes.yaml
    oligo_parts: catalogs/oligo_parts.yaml
    backbones: catalogs/backbones.yaml
  output:
    run_dir: outputs/yiu/explicit
    publish_contract_version: 2
```

### Required concepts

- `source_oligo.sequence`: concrete or IUPAC DNA sequence for the authored source oligo
- `schema_version: 2` makes YIU a protocol-template family. `msd_hop_retron_eco1_v1` models the concrete msd-HOP/Y-adapter hairpin workflow rather than the older generic circularization lane.
- explicit reports distinguish `sequence_mode` from `validation_mode`: a `v1` spec with active IUPAC ambiguity reports `iupac_pattern`, while `v2` reports `pattern`; both still surface `pattern_compatibility` instead of `concrete_realization`
- `step_graph`: fixed phase-1 graph from `pcr` through `amplification`
- `steps`: the `v2` template-correct graph from `source_pcr` through `hairpin_pcr`, with optional `insert_cleanup`, `backbone_pcr`, and `golden_gate_assembly`
- `assembled_payload`: the target retained payload after ligation/junction assembly
- `assembled_payload_pattern`: the `v2` payload goal checked in assembled coordinate space
- `payload_goal.left_half_ref` / `payload_goal.right_half_ref`: named payload windows on the source oligo
- `catalogs`: optional workspace-relative protocol catalogs for restriction enzymes, nickases, adapters, and `v2` oligo/backbone parts
- `adapter_ligation`: the adapter sequence may come from `step.adapter_sequence`, `adapter_policy.adapter_sequence`, or `adapter_policy.y_adapter_id` plus `catalogs.adapters`
- `protocol_template: msd_hop_retron_eco1_v1` uses typed intermediate states: `source_amplicon_dsdna`, `post_double_nicking_fragment_pool`, `post_heat_cleanup_fragment_pool`, `adapter_annealed_complex`, `ligated_ssdna_hairpin`, and `hairpin_pcr_linear_insert`
- `circularization`: `exact_complement`, `partial_complement`, and `bulged` are now distinct deterministic sticky-end geometry modes
  `min_paired_nt` sets the minimum reverse-complement core size required by `partial_complement` and `bulged`.
  `max_unpaired_tail_nt` bounds the allowed slack at the overhang ends.
  `max_bulge_nt` bounds the single internal insertion/deletion allowed only in `bulged`.
- `adapter_anneal` and `hairpin_ligation` in `v2` use the same three-way ligation compatibility model, but on typed annealed/hairpin states instead of the legacy circularization noun
- `publish_contract_version: 2` enables the additive typed published view contract
- `evidence_policy`: `require_guaranteed` rejects merely possible IUPAC matches; `allow_possible_with_warning` keeps the run satisfied but emits warnings
- primer annotations are split into `primer_binding_cores` and `primer_tails` so non-annealing handles do not silently claim source-coordinate overlap
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

`v2` oligo-parts catalog:

```yaml
oligo_parts:
  entries:
    - id: oES790
      part_kind: primer
      sequence: GGAAAA
    - id: oES792
      part_kind: adapter
      sequence: ACCGGTTAA
      phosphorylated_5p: true
```

`v2` backbone catalog:

```yaml
backbones:
  entries:
    - id: demo_backbone
      sequence: GGTCTC...
```

### Important invariants

- `schema_version: 1` stays accepted and keeps its current state IDs and semantics
- `schema_version: 2` gains `protocol_template`, `workflow_scope`, typed topology kinds, additive view contract `2`, and honest pattern-evidence accounting
- annotation ids must be unique across primer sites, enzyme sites, payload windows, homology windows, retained regions, and sacrificial regions
- `v2` `source_oligo.annotations` also requires unique ids across primer tails and overlap overrides
- the PCR step records `amplicon_start`, `amplicon_end`, and `amplicon_length_nt`, and emits a structured issue if downstream annotations fall outside that primer-defined amplicon
- the restriction-digest state publishes the cut-retained duplex view rather than the full PCR product, including explicit cut boundaries, removed primary flanks, and projected annotation coordinates relative to the digested state
- retained and sacrificial regions are checked explicitly: any overlap emits `RETAINED_SACRIFICIAL_OVERLAP`, and partially overlapping retained-only or sacrificial-only regions are rejected before the protocol trace runs
- `v2` also rejects `PAYLOAD_SACRIFICIAL_OVERLAP` and `PRIMER_CORE_SITE_OVERLAP`
- `v2` closes the IUPAC honesty gap: each state carries `pattern_evidence_summary` and each required check resolves to `guaranteed`, `possible`, or `impossible`
- `v2` published states keep `primary_sequence`, `complement_sequence`, and `meta`, but add `view_contract_version`, `state_kind`, `topology_kind`, `segments`, `annotations`, `cuts`, `junctions`, and `fragments`
- circularization reports now publish sticky-end compatibility geometry explicitly: `paired_nt`, `unpaired_tail_nt`, `bulge_nt`, `bulge_side`, and aligned core coordinates in addition to the legacy `sticky_end_overlap`
- the circularization state publishes `payload_junction_segments` and `payload_junction`, making the source-to-assembled-payload join explicit under `payload_goal.junction_rule`
- restriction and nickase sites must match the source oligo in the requested orientation
- each sacrificial region referenced by `nickase_digest.sacrificial_region_ids` must actually be cut by at least one configured nickase site or the workflow emits `NICKASE_SACRIFICIAL_REGION_UNCUT`
- the nickase-fragmentation state publishes retained component mappings explicitly, including source coordinates and retained-product coordinates for each retained region
- foldback homology windows are validated against the retained-product state rather than raw source coordinates; a window that is no longer preserved emits `HOMOLOGY_WINDOW_EXCLUDED_FROM_CURRENT_STATE`
- projected foldback windows now use explicit region projections. Each projected window carries `parts[]` plus `spans_junction`; a junction-spanning window is represented in the report and rejected with `HOMOLOGY_WINDOW_SPANS_JUNCTION`
- `v2` payload assembly across a retained-product junction is represented explicitly as a compound projection in assembled coordinate space instead of flattening back to one source interval
- the adapter-ligated state publishes explicit branched-Y metadata: `topology: branched_y`, ordered `arms`, and `branch_junction`
- referenced catalog entries must exist and match the authored enzyme motifs, cut offsets, and adapter sequences
- `step_graph.steps` must use the fixed YIU order for the explicit tracer bullet
- `output.run_dir` must stay inside the workspace root

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Artifacts](yiu_artifacts.md) for the emitted bundle contract.
