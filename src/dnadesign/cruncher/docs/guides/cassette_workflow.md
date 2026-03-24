## Cassette workflow

**Owner:** dnadesign-maintainers
**Doc kind:** guide
**Audience:** cassette workflow users and maintainers
**Updated by:** cruncher-maintainers on 2026-03-24
**Applies to:** `uv run cruncher cassette validate|design|show`
**Last verified:** 2026-03-24
**Primary artifacts:** `analysis/reports/report.json`, `analysis/reports/report.md`, `analysis/reports/render_contract.json`, `export/table__candidates.csv`

### Contents
- [Why this exists](#why-this-exists)
- [Current workflow scope](#current-workflow-scope)
- [Workspace layout](#workspace-layout)
- [Minimal catalog](#minimal-catalog)
- [Minimal spec](#minimal-spec)
- [Standard command sequence](#standard-command-sequence)
- [Outputs](#outputs)
- [Failure modes](#failure-modes)

### Why this exists

`sample` optimizes fixed-length PWM-driven sequences. `cassette` is a separate workflow for an explicitly authored
dual-context design object:

- in ssDNA context, the cassette forms a stem-loop hairpin
- in linear dsDNA context, the same cassette yields two intended strand-specific nick calls on one designated strand

The workflow is strict about reporting a `bounded_segment` between intended nicks. It does not claim that the segment is removed.

### Current workflow scope

Current v1 behavior:

- validate a strict cassette spec at `configs/cassettes/<name>.cassette.yaml`
- validate a strict local nickase catalog
- derive the 3' arm as the reverse complement of the 5' arm
- scan the duplex context for the requested left/right nickase sites
- emit a deterministic satisfied or unsatisfied report
- write cassette-specific artifacts under `outputs/cassettes/`

Current v1 non-scope:

- no generalized search over stem sequence, loop sequence, or nickase choice
- no remote nickase registry fetch
- no downstream excision/removal simulation

This workflow does not currently search over stems, loops, or nickase assignments.

Use this workflow when you already know the intended cassette topology and nickase assignments and need Cruncher to validate
the dual-context contract and publish artifacts.

### Workspace layout

Store the spec and catalog inside the workspace:

```text
<workspace>/
  configs/
    cassettes/
      demo_hairpin.cassette.yaml
  inputs/
    nickases/
      demo.nickases.yaml
  outputs/
    cassettes/
```

`--spec` must point to the concrete `.cassette.yaml` file. Passing `configs/` or a file outside the workspace `configs/` tree fails fast.

### Minimal catalog

```yaml
nickases:
  schema_version: 1
  entries:
    - id: nb_left
      recognition_sequence: AACGA
      nicked_site_strand: forward
      cut_offset: 2
    - id: nb_right
      recognition_sequence: AACGA
      nicked_site_strand: reverse
      cut_offset: 3
```

See [`../reference/nickase_catalog.md`](../reference/nickase_catalog.md) for full field semantics.

### Minimal spec

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

This example validates to one cassette sequence, `AACGATTTATCGTT`, with reported nick coordinates `2` and `12`.

See [`../reference/cassette_spec.md`](../reference/cassette_spec.md) for the full schema.

### Standard command sequence

```bash
set -euo pipefail

# 1) Validate the spec and inspect the report.
uv run cruncher cassette validate --spec configs/cassettes/demo_hairpin.cassette.yaml
uv run cruncher cassette validate --spec configs/cassettes/demo_hairpin.cassette.yaml --json

# 2) Materialize a deterministic cassette run directory.
uv run cruncher cassette design --spec configs/cassettes/demo_hairpin.cassette.yaml

# 3) Inspect the resulting paths.
uv run cruncher cassette show --run outputs/cassettes/demo_hairpin/<design_id>
```

Notes:

- `validate` exits nonzero for unsatisfied specs.
- `design` still writes a cassette run directory for unsatisfied specs so the issue report is preserved.
- `show` reads cassette-specific metadata only; it does not use `cruncher runs ...`.

### Outputs

Cassette outputs are deterministic:

```text
<workspace>/outputs/cassettes/<spec.name>/<design_id>/
```

Primary artifacts:

- `meta/cassette_manifest.json`
- `meta/cassette_status.json`
- `provenance/spec_used.yaml`
- `provenance/nickase_catalog.yaml`
- `analysis/reports/report.json`
- `analysis/reports/report.md`
- `analysis/reports/render_contract.json` when `output.write_render_contract: true`
- `export/table__candidates.csv`

The render contract publishes two views:

- `ssdna_hairpin`
- `linear_duplex`

See [`../reference/cassette_artifacts.md`](../reference/cassette_artifacts.md) for file-by-file semantics.

### Failure modes

Common fail-fast cases:

- spec path is not `configs/cassettes/<name>.cassette.yaml`
- `catalog.path` escapes the workspace or the catalog file is missing
- nickase IDs referenced in the spec are absent from the catalog
- no designated-strand nick lands inside the requested left or right window
- multiple candidate nick calls land inside one requested window
- the left nick is not strictly before the right nick
- additional designated-strand nick sites exist when `forbid_additional_designated_strand_nicks: true`

When no valid pair exists, Cruncher reports explicit issue codes such as `missing_right_nick`, `ambiguous_left_nick`,
or `extra_designated_strand_nicks`. There is no fallback to `sample`.
