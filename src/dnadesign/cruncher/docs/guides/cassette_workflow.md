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

`sample` optimizes fixed-length PWM-driven sequences. `cassette` is a separate workflow for an explicitly authored dual-context design object:

- in ssDNA context, the cassette forms a stem-loop hairpin
- in linear dsDNA context, the same cassette yields two intended strand-specific nick calls on one target strand

The workflow is strict about reporting a `bounded_nicked_segment` between intended nicks. It does not claim that the segment is removed.

### Current workflow scope

This guide covers the explicit cassette lane only: `validate`, `design`, and `show`.

Current explicit-lane behavior:

- validate a strict cassette spec at `configs/cassettes/<name>.cassette.yaml`
- validate a strict local nickase catalog
- derive the 3' arm as the reverse complement of the 5' arm
- scan the evaluated duplex context for requested or catalog-scoped nickase instances
- emit a deterministic satisfied or unsatisfied report with stable issue codes
- write cassette-specific artifacts under `outputs/cassettes/`

Current non-scope:

- no generalized search inside `validate|design|show`; use `cassette solve` for search-backed design
- no remote nickase registry fetch
- no downstream excision/removal simulation

The explicit lane does not search over stems, loops, or nickase assignments.

Use this workflow when you already know the intended cassette topology and nickase assignments and need Cruncher to validate the dual-context contract and publish artifacts.

If you want Cruncher to search over patterned stems/loops and rank multiple hits, use
[`cassette_solve_workflow.md`](cassette_solve_workflow.md) instead.

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
    - id: Nt.demo
      specificity_id: Demo
      motif_top_5to3: AACGA
      top_cut_offset: 2
    - id: Nb.demo
      specificity_id: Demo
      motif_top_5to3: AACGA
      bottom_cut_offset: 2
```

See [`../reference/nickase_catalog.md`](../reference/nickase_catalog.md) for full field semantics.

### Minimal spec

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
      nick_window:
        start: 2
        end: 2
    right:
      nickase: Nb.demo
      nick_window:
        start: 12
        end: 12
    require_exactly_two_intended_nicks: true
    bounded_segment_length:
      min: 10
      max: 10
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

This example validates to one cassette sequence, `AACGATTTATCGTT`, with intended boundaries `2` and `12`.

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
- Unsupported tracer-bullet mode switches fail fast at load time. Today that includes `require_exactly_two_intended_nicks: false`, `require_topological_hairpin: false`, and `require_energetic_hairpin: true`.

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
- no intended boundary lands inside the requested left or right window
- intended site spans the loop or a stem boundary
- the intended window only matches the opposite target strand
- the left boundary is not strictly before the right boundary
- additional designated-strand nick sites exist when `forbid_additional_designated_strand_nicks: true`

When no valid pair exists, Cruncher reports explicit issue codes such as `RIGHT_WINDOW_NO_MATCH`,
`TARGET_STRAND_MISMATCH`, `RIGHT_SITE_NOT_IN_RIGHT_STEM`, `EXTRA_DESIGNATED_STRAND_NICKS_FOUND`,
or `UNSAT_BY_MIRROR_SYMMETRY`. There is no fallback to `sample`.
