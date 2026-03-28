## YIU Spec Reference

**Audience:** YIU workflow users and maintainers
**Applies to:** `configs/yiu/*.yiu.yaml` and `configs/yiu/*.yiu.solve.yaml`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-27

YIU now has two strict document roots:

- explicit specs rooted at `yiu:`
- solve specs rooted at `yiu_solve:`

### Recommended workspace layout

```text
configs/
  runbook.yaml
  yiu/
    example_split_payload_circularized.yiu.yaml
    example_split_payload_circularized.yiu.solve.yaml
    compat/
      example_adapter_hairpin.yiu.yaml
      example_legacy_v1.yiu.yaml
catalogs/
  enzymes.yaml
  oligo_parts.yaml
  backbones.yaml
outputs/
  yiu/
    explicit/
    solve/
```

### Explicit spec summary

The explicit root is still `yiu:`. Three shapes matter operationally:

- `schema_version: 1` with `protocol: yiu_v1`
- `schema_version: 2` with `protocol_template: yiu_adapter_hairpin_v1`
- `schema_version: 2` with `protocol_template: yiu_circularized_payload_v1`

Recommended split-template explicit spec:

```yaml
yiu:
  schema_version: 2
  family: yiu
  protocol_template: yiu_circularized_payload_v1
  workflow_scope: core_insert_generation
  name: example_split_payload_circularized
  source_oligo:
    authored_sequence: CCGATG...
    annotations:
      primer_binding_cores: [...]
      restriction_sites: [...]
      nickase_sites: [...]
      payload_windows: [...]
      retained_regions: [...]
      sacrificial_regions: [...]
      named_regions: [...]
  steps:
    source_pcr: {...}
    type_iis_digest: {...}
    circularization: {...}
    exonuclease_cleanup: {...}
    sacrificial_digest: {...}
    fragment_cleanup: {...}
    snapback_adapter_engagement: {...}
    hairpin_ligation: {...}
    hairpin_pcr: {...}
  payload_goal:
    assembled_payload_pattern: TCCCTATCAGTGATAGAGA
    left_half_ref: payload_left
    right_half_ref: payload_right
    assembly_space: circularized_payload_junction
    evidence_policy: require_guaranteed
  template_bindings:
    source_forward_primer_core_ref: source_fwd_core
    source_reverse_primer_core_ref: source_rev_core
    snapback_seed_region_ref: snapback_seed
    retained_left_region_ref: retained_payload_left
    retained_right_region_ref: retained_payload_right
    primary_sacrificial_region_refs:
      - sacrificial_tract
    circularization_left_overhang_ref: split_payload_digest
    circularization_right_overhang_ref: split_payload_digest
  compound_regions:
    - id: assembled_payload
      join_policy: junction_assemble
      segments: [...]
  hard_invariants:
    - id: payload_assembly
      class: payload_assembly
      space_kind: assembly_junction
      transform_ref: circularization
      region_ref: assembled_payload
      evidence_policy: require_guaranteed
      params:
        expected_pattern: TCCCTATCAGTGATAGAGA
  catalogs:
    enzymes: catalogs/enzymes.yaml
    oligo_parts: catalogs/oligo_parts.yaml
    backbones: catalogs/backbones.yaml
  output:
    run_dir: outputs/yiu/explicit
    emit_view_contracts: true
    emit_baserender_jobs: true
    publish_contract_version: 3
```

Important explicit rules:

- `publish_contract_version: 3` is the default for `schema_version: 2`
- `publish_contract_version: 2` is compatibility-only for older consumers
- `output.emit_baserender_jobs` requires `output.emit_view_contracts: true`
- `template_bindings` is required for `protocol_template: yiu_circularized_payload_v1`
- sequence identity fields must be separator-free; composite structure belongs in `segments`, `junctions`, `compound_regions`, and optional display metadata
- hard invariants are either evaluated or rejected; they are no longer silently accepted
- `cloning_geometry` is not allowed in `workflow_scope: core_insert_generation`

### Hard invariants

The shipped split-template lane uses `payload_assembly`, `sacrificial_fragmentation`, and `snapback_exposure`, and the schema still recognizes:

- `region_pattern`
- `enzyme_site`
- `cut_geometry`
- `ligation_compatibility`
- `payload_assembly`
- `retained_survival`
- `sacrificial_fragmentation`
- `snapback_exposure`
- `adapter_binding`
- `primer_binding`
- `cloning_geometry`

Every invariant result records:

- `class`
- evaluated state or transform
- `status`: `guaranteed`, `possible`, or `impossible`
- observed coordinates, motifs, fragments, or junction evidence

### Workflow support matrix

Current shipment language:

> YIU currently ships as a split-payload circularized workflow family with bounded solve over declared source windows. Compatibility templates remain supported, but the split-payload circularized template is the recommended operator path.

| Surface | `yiu_circularized_payload_v1` | `yiu_adapter_hairpin_v1` | `yiu_v1` |
| --- | --- | --- | --- |
| `validate/design/trace/show` | full | compatibility-supported | compatibility-supported |
| `solve` | full bounded support | not supported | not supported |
| visual publication | full v3 support | compatibility / reduced guarantees | compatibility / reduced guarantees |
| BaseRender jobs | supported path | compatibility path | not emitted |

Runtime-supported invariant classes for the split-payload lane:

| Invariant class | Split-payload explicit validate | Split-payload solve admissibility | Notes |
| --- | --- | --- | --- |
| `payload_assembly` | yes | yes | compound-region aware |
| `sacrificial_fragmentation` | yes | yes | bounded fragment check |
| `snapback_exposure` | yes | yes | 5' exposure-aware |
| `retained_survival` | yes | yes | projected-region aware |
| `ligation_compatibility` | yes | yes | step-match backed |
| `enzyme_site` | yes | yes | site projection required |
| `cut_geometry` | yes | yes | site projection required |
| `adapter_binding` | yes | yes | engagement/ligation backed |
| `primer_binding` | yes | yes | binding-core presence check |
| `region_pattern` | yes | yes | generic region / compound-region check |
| `cloning_geometry` | rejected | rejected | not implemented in this tranche |

### Solve spec summary

The solve root is `yiu_solve:` and must live beside the explicit specs under `configs/yiu/`.

```yaml
yiu_solve:
  schema_version: 1
  base_spec: configs/yiu/example_split_payload_circularized.yiu.yaml
  search:
    max_hits: 32
    materialize_top_k: 8
    max_search_nodes: 100000
    max_enumerated_candidates: 10000
  variables:
    source_windows:
      - id: payload_left
        span_ref: payload_left
        alphabet: iupac_dna
        pattern: NNNNNNNNN
      - id: payload_right
        span_ref: payload_right
        alphabet: iupac_dna
        allowed_patterns:
          - GTGATAGAGA
  candidate_policy:
    require_guaranteed_hard_invariants: true
    forbid_possible_hits: true
  output:
    run_dir: outputs/yiu/solve
    emit_view_contracts: true
    emit_baserender_jobs: true
    publish_contract_version: 3
```

Important solve rules:

- `base_spec` is resolved relative to the workspace root
- solve currently targets `schema_version: 2` base specs
- variable windows must match the referenced span length exactly
- solve hits are concrete and bounded; search is capped by both `max_search_nodes` and `max_enumerated_candidates`
- ranking never weakens admissibility; the explicit validator remains the final oracle

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Artifacts](yiu_artifacts.md) for the emitted bundle contracts.
