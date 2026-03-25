## Cassette spec reference

**Owner:** dnadesign-maintainers
**Doc kind:** reference
**Audience:** cassette workflow users and maintainers
**Updated by:** cruncher-maintainers on 2026-03-24
**Applies to:** `configs/cassettes/*.cassette.yaml`
**Last verified:** 2026-03-24
**Primary artifacts:** validated `report.json`, `report.md`, optional `render_contract.json`

### Contents
- [File location](#file-location)
- [Schema versions](#schema-versions)
- [Canonical v2 shape](#canonical-v2-shape)
- [Field semantics](#field-semantics)
- [Coordinate semantics](#coordinate-semantics)
- [Hard invariants](#hard-invariants)
- [Backward-compatible aliases](#backward-compatible-aliases)

### File location

Cassette specs must live at:

```text
<workspace>/configs/cassettes/<name>.cassette.yaml
```

The CLI rejects directory arguments, files outside the workspace `configs/` tree, and paths with `..` traversal.

Solve-mode search specs use the separate suffix `.cassette.solve.yaml` and are documented in
[`cassette_solve_spec.md`](cassette_solve_spec.md).

### Schema versions

- `schema_version: 2` is the canonical tracer-bullet schema.
- `schema_version: 1` is still accepted and preserves legacy window semantics.
- `schema_version` is required. The loader does not guess.

### Canonical v2 shape

```yaml
cassette:
  schema_version: 2
  name: demo_hairpin
  topology:
    stem5p_arm: AACGAT
    loop: TT
    stem3p_arm_mode: derived_reverse_complement
  construct_context:
    left_flank: ""
    right_flank: ""
  nicking:
    target_strand: primary
    left:
      nickase: Nt.demo
      nick_window: {start: 2, end: 2}
    right:
      nickase: Nb.demo
      nick_window: {start: 12, end: 12}
    require_exactly_two_intended_nicks: true
    bounded_segment_length: {min: 10, max: 10}
  site_policy:
    forbid_additional_designated_strand_nicks: false
    scan_scope: requested_variants
  hairpin_validation:
    require_topological_hairpin: true
    require_energetic_hairpin: false
  catalog:
    path: inputs/nickases/demo.nickases.yaml
  output:
    run_dir: outputs/cassettes
    write_render_contract: true
```

### Field semantics

- `schema_version`: `1` or `2`.
- `name`: non-empty run namespace. It becomes part of `outputs/cassettes/<name>/...`.
- `topology.stem5p_arm`: required concrete DNA sequence (`A/C/G/T` only).
- `topology.loop`: required concrete DNA sequence (`A/C/G/T` only).
- `topology.stem3p_arm_mode`: must resolve to `derived_reverse_complement`. The tracer bullet does not support authoring `stem3p_arm` directly.
- `construct_context.left_flank`, `construct_context.right_flank`: optional concrete flanks included when evaluating duplex nickase instances.
- `nicking.target_strand`: duplex strand that must receive both intended nicks: `primary` or `complement`.
- `nicking.left.nickase`, `nicking.right.nickase`: nickase variant IDs that must exist in the referenced local catalog.
- `nicking.left.nick_window`, `nicking.right.nick_window`: cassette-local intended nick boundary windows.
- `nicking.require_exactly_two_intended_nicks`: must be `true` in the current tracer bullet. Multi-intended modes are outside scope and fail fast at load time.
- `nicking.bounded_segment_length`: optional inclusive length interval for the bounded nicked segment.
- `site_policy.forbid_additional_designated_strand_nicks`: when `true`, extra designated-strand nick events under the active scan scope produce an unsatisfied report.
- `site_policy.scan_scope`: `requested_variants` or `catalog`.
- `hairpin_validation.require_topological_hairpin`: must be `true` in the current tracer bullet. Disabling the topological hairpin contract is not supported.
- `hairpin_validation.require_energetic_hairpin`: reserved for future energetic checks. Setting it to `true` currently fails fast instead of silently skipping thermodynamic validation.
- `catalog.path`: workspace-relative or absolute path to a local nickase catalog.
- `output.run_dir`: relative output root inside the workspace. Absolute paths and `..` traversal are rejected.
- `output.write_render_contract`: when `true`, write and report `analysis/reports/render_contract.json`.

### Coordinate semantics

- Internal cassette planning uses bond-boundary coordinates.
- `schema_version: 2` reports `coordinate_semantics: boundary_inclusive_v2`.
- In `boundary_inclusive_v2`, cassette-local legal nick boundaries are `0..N`, where `0` is before the first cassette base and `N` is after the last cassette base.
- `nick_window.start` and `nick_window.end` are inclusive boundary windows.
- `RecognitionSiteInstance.start` and `.end` in reports are positions on the evaluated primary display strand.
- `NickEvent.boundary` is cassette-local.
- `NickEvent.boundary_context` is the same nick boundary projected into the evaluated primary sequence.
- `schema_version: 1` preserves the legacy v1 interpretation and reports `coordinate_semantics: legacy_v1`.

### Hard invariants

- The cassette primary strand is always `stem5p_arm + loop + reverse_complement(stem5p_arm)`.
- The tracer bullet models exactly one loop, no bulges, and no mismatches.
- Intended left and right recognition-site instances must each lie wholly inside their respective stem arms.
- Both intended nick events must land on the requested `target_strand`.
- The left intended boundary must be strictly less than the right intended boundary.
- The output reports a **bounded nicked segment**, not excision.
- Nickase IDs must resolve against the referenced catalog before planning begins.

### Backward-compatible aliases

The loader accepts these aliases and normalizes them internally:

- `duplex_context.upstream` -> `construct_context.left_flank`
- `duplex_context.downstream` -> `construct_context.right_flank`
- `nicking.designated_strand` -> `nicking.target_strand`
- `topology.stem3p_arm_mode: derive_reverse_complement` -> `derived_reverse_complement`

If both old and new names are present in the same document, the loader fails with `SCHEMA_ALIAS_CONFLICT`.
