## YIU Spec Reference

**Audience:** YIU workflow users and maintainers
**Applies to:** `configs/yiu/*.yiu.yaml` and `configs/yiu/*.yiu.solve.yaml`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-29

YIU ships two strict document roots:

- explicit specs rooted at `yiu:`
- solve specs rooted at `yiu_solve:`

### Recommended workspace layout

```text
configs/
  runbook.yaml
  yiu/
    <workflow>.yiu.yaml
    <workflow>.yiu.solve.yaml
catalogs/
  enzymes.yaml
  oligo_parts.yaml
  backbones.yaml
outputs/
  yiu/
    explicit/
    solve/
```

### Explicit spec

The explicit root is `yiu:` and the ship-targeted contract is `schema_version: 4`.

```yaml
yiu:
  schema_version: 4
  family: yiu
  protocol_template: yiu_circularized_payload_v1
  name: <workflow>
  source_oligo:
    authored_sequence: <iupac_dna_sequence>
    structural_owners:
      - id: source_fwd_primer_binding_region
        start: 0
        end: 6
      - id: payload_left_half
        start: 6
        end: 15
      - id: sacrificial_region_long
        start: 15
        end: 27
      - id: tether_dock_complement
        start: 27
        end: 31
      - id: tether_cap
        start: 31
        end: 35
      - id: tether_dock
        start: 35
        end: 39
      - id: snapback_stem
        start: 39
        end: 41
      - id: payload_right_half
        start: 41
        end: 51
      - id: source_rev_primer_binding_region
        start: 51
        end: 57
    effect_tags:
      - id: source_forward_primer_bindable
        class: primer_bindable_by_source_forward
        start: 0
        end: 6
      - id: left_bsssi_bsai_overlap_unit
        class: left_bsssi_bsai_overlap_unit
        start: 0
        end: 18
      - id: payload_overhang_left
        class: payload_overhang_left
        start: 6
        end: 10
      - id: type_iis_recognition_left
        class: type_iis_recognition_left
        start: 7
        end: 13
      - id: nt_bpu10i_snapback_site
        class: nt_bpu10i_snapback_site
        start: 27
        end: 41
      - id: payload_overhang_right
        class: payload_overhang_right
        start: 41
        end: 45
      - id: type_iis_recognition_right
        class: type_iis_recognition_right
        start: 51
        end: 57
  owner_lifecycle:
    - owner_id: payload_left_half
      appears_in:
        - source_oligo_ssdna
      projected_to:
        - state: pcr_linear_duplex
          strand: primary
          provenance_mode: literal_source
        - state: type_iis_cut_product_duplex
          strand: primary
          provenance_mode: cut_product_projection
        - state: circularized_payload_candidate
          strand: primary
          provenance_mode: ligation_assembly
        - state: post_fragment_cleanup
          strand: primary
          provenance_mode: retained_projection
        - state: hairpin_pcr_linear_insert
          strand: primary
          provenance_mode: amplification_projection
      disappears_after: null
  external_parts:
    primer_source_forward: <oligo_id>
    primer_source_reverse: <oligo_id>
    hairpin_pcr_forward: <oligo_id>
    hairpin_pcr_reverse: <oligo_id>
    y_adapter: <oligo_id>
  enzymes:
    left_type_iis: BsmBI
    right_type_iis: BsmBI
    snapback_nickase: Nt.Bpu10I
    sacrificial_nickase: Nb.BssSI
  payload:
    target_sequence: <assembled_payload_sequence>
    bulge_mask: []
  catalogs:
    enzymes: catalogs/enzymes.yaml
    oligo_parts: catalogs/oligo_parts.yaml
    backbones: catalogs/backbones.yaml
  output:
    run_dir: outputs/yiu/explicit
    emit_view_contracts: true
    publish_contract_version: 4
    persist_render_jobs_debug: false
```

Required structural-owner ids for `yiu_circularized_payload_v1`:

- `source_fwd_primer_binding_region`
- `payload_left_half`
- `sacrificial_region_long`
- `tether_dock_complement`
- `tether_cap`
- `tether_dock`
- `snapback_stem`
- `payload_right_half`
- `source_rev_primer_binding_region`
- `retained_region`
- `sacrificial_region_short`
- `y_adapter_complementary_arm`
- `y_adapter_noncomplementary_arm`
- `hairpin_pcr_forward_binding_region`
- `hairpin_pcr_reverse_binding_region`

Closed effect-tag kinds:

- `type_iis_recognition_left`
- `type_iis_recognition_right`
- `payload_overhang_left`
- `payload_overhang_right`
- `nt_bpu10i_snapback_site`
- `nb_bsssi_array_member`
- `left_bsssi_bsai_overlap_unit`
- `pairs_with`
- `primer_bindable_by_source_forward`
- `primer_bindable_by_source_reverse`
- `primer_bindable_by_hairpin_pcr_forward`
- `primer_bindable_by_hairpin_pcr_reverse`
- `retained`
- `sacrificial`
- `introduced_late`
- `y_adapter_binding`
- `ligation_junction_member`
- `cut_boundary_anchor`
- `nick_boundary_anchor`
- `payload_bulge_position`

Emitted state ids:

1. `source_oligo_ssdna`
2. `pcr_linear_duplex`
3. `type_iis_cut_product_duplex`
4. `circularized_payload_candidate`
5. `post_sacrificial_fragmentation`
6. `post_fragment_cleanup`
7. `snapback_adapter_complex`
8. `ligated_ssdna_hairpin`
9. `hairpin_pcr_linear_insert`

Important explicit rules:

- structural owners must cover the entire authored source sequence in the declared source order
- every emitted nucleotide belongs to exactly one structural owner on each emitted strand
- unknown effect-tag kinds hard-fail at load time
- owner and effect-tag overlap legality fails closed
- runtime state-owner behavior must come from `owner_lifecycle`
- `projection_mode`, `compound_regions`, `join_policy`, and `space_kind` are not part of the v4 operational schema

### Solve spec

The solve root is `yiu_solve:` and lives beside the explicit spec under `configs/yiu/`.

```yaml
yiu_solve:
  schema_version: 1
  base_spec: configs/yiu/<workflow>.yiu.yaml
  target:
    payload_sequence: <assembled_payload_sequence>   # or payload_pattern
    bulge_mask: []
  scaffold_windows:
    - id: sacrificial_spacing_window
      owner_id: sacrificial_region_long
      relative_start: 3
      relative_end: 12
      allowed_patterns:
        - AAAAAAAAA
        - AAAATAAAA
  search:
    max_search_nodes: 16
    max_enumerated_candidates: 16
  solve:
    compare_solutions: false
    max_solutions: 1
  output:
    run_dir: outputs/yiu/solve
    emit_view_contracts: true
    publish_contract_version: 4
    persist_render_jobs_debug: false
```

Important solve rules:

- `base_spec` is resolved relative to the solve spec path
- payload halves and payload overhangs are derived from one payload target
- only declared `scaffold_windows` may mutate fixed scaffold sequence
- solve windows must map to structural-owner spans
- default solve returns one deterministic solution, not a ranked hit list
- `compare_solutions: true` enables additional solutions and `comparison/solutions.csv`
- public solve statuses are `solved`, `unsatisfied`, and `incomplete_search`

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Artifacts](yiu_artifacts.md) for the emitted bundle contracts.
