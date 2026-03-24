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
- [Top-level shape](#top-level-shape)
- [Field semantics](#field-semantics)
- [Coordinate semantics](#coordinate-semantics)
- [Hard invariants](#hard-invariants)
- [Example](#example)

### File location

Cassette specs must live at:

```text
<workspace>/configs/cassettes/<name>.cassette.yaml
```

The CLI rejects directory arguments, files outside the workspace `configs/` tree, and paths with `..` traversal.

### Top-level shape

```yaml
cassette:
  schema_version: 1
  name: demo_hairpin
  topology:
    stem5p_arm: AACGAT
    loop: TT
    stem3p_arm_mode: derive_reverse_complement
  duplex_context:
    upstream: ""
    downstream: ""
  nicking:
    designated_strand: primary_strand
    left:
      nickase: nb_left
      nick_window: {start: 0, end: 3}
    right:
      nickase: nb_right
      nick_window: {start: 11, end: 13}
    forbid_additional_designated_strand_nicks: false
  catalog:
    path: inputs/nickases/demo.nickases.yaml
  output:
    run_dir: outputs/cassettes
    write_render_contract: true
```

### Field semantics

- `schema_version`: must be `1`.
- `name`: non-empty run namespace. It becomes part of `outputs/cassettes/<name>/...`.
- `topology.stem5p_arm`: required DNA sequence (`A/C/G/T` only).
- `topology.loop`: required DNA sequence (`A/C/G/T` only).
- `topology.stem3p_arm_mode`: must be `derive_reverse_complement` in v1.
- `duplex_context.upstream`, `duplex_context.downstream`: optional flanking DNA included when scanning nickase recognition sites.
- `nicking.designated_strand`: which duplex strand must receive both intended nicks: `primary_strand` or `complement_strand`.
- `nicking.left.nickase`, `nicking.right.nickase`: nickase IDs that must exist in the referenced catalog.
- `nicking.left.nick_window`, `nicking.right.nick_window`: cassette-relative windows matched against reported `nick_coordinate` values.
- `nicking.forbid_additional_designated_strand_nicks`: when `true`, any extra designated-strand nick site causes an unsatisfied report.
- `catalog.path`: workspace-relative or absolute path to a local nickase catalog.
- `output.run_dir`: relative output root inside the workspace. Absolute paths and `..` traversal are rejected.
- `output.write_render_contract`: when `true`, write and report `analysis/reports/render_contract.json`.

### Coordinate semantics

- `nick_window.start` and `nick_window.end` are zero-based inclusive cassette coordinates.
- `PlannedNick.site_start` and `PlannedNick.site_end` in reports are cassette-relative half-open recognition-site spans.
- `nick_coordinate_context` in reports is the same nick coordinate projected into the duplex context sequence.
- `bounded_segment.start` and `bounded_segment.end` come directly from the selected left/right `nick_coordinate` values.

The workflow uses the same reported `nick_coordinate` integers for filtering windows, reports, and the optional render contract.

### Hard invariants

- `stem3p_arm` is always derived as `reverse_complement(stem5p_arm)`.
- both intended nick calls must land on the requested `designated_strand`.
- the left nick coordinate must be strictly less than the right nick coordinate.
- the bounded segment is reported as a nick-bounded interval, not as an excised/removed product.
- nickase IDs must resolve against the referenced catalog before planning begins.

### Example

```yaml
cassette:
  schema_version: 1
  name: demo_hairpin
  topology:
    stem5p_arm: AACGAT
    loop: TT
    stem3p_arm_mode: derive_reverse_complement
  duplex_context:
    upstream: ""
    downstream: ""
  nicking:
    designated_strand: primary_strand
    left:
      nickase: nb_left
      nick_window:
        start: 0
        end: 3
    right:
      nickase: nb_right
      nick_window:
        start: 11
        end: 13
  catalog:
    path: inputs/nickases/demo.nickases.yaml
  output:
    run_dir: outputs/cassettes
    write_render_contract: true
```
