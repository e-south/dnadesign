# MSD Genetic Compiler

Use this reference when a user provides a Retron MSD shorthand ID or enough
parts to compile a design reference.

## Input Completeness

Minimum complete reference:

- payload or target id
- cap id
- left base
- right base
- optional profile code, which must match the computed `S3/S2/S1/S0` profile

If the user provides a full lab-facing label, parse those fields from the label.
If any primitive is missing, route instead of guessing.

## Boundary

- Study-owned package: `dnadesign.studies.retron_hairpin_design`.
- Registry: `docs/studies/retron_hairpin_design/msd_design_registry.yaml`.
- Selected labels: `docs/studies/retron_hairpin_design/msd_design_hit_labels.txt`.
- Do not expose a top-level `retron-msd` script.
- Do not create Construct or Folding workspaces per ID.
- Snapback and scar-nick solve missing primitive parts; the compiler validates
  selected parts into reference contracts and one MSD unit sequence per design.
- A materialized MSD unit is 5' flank + left base, payload primary, cap
  geometry, payload complement, right base + 3' flank.
- Do not add a repeat count or chain complete MSD units together.
- Reader consumes `msd_design_catalog_v1`, not live dnadesign internals.

## Commands

Lint one label:

```bash
uv run python -m dnadesign.studies.retron_hairpin_design.cli lint \
  --id "pES-retron-177-msd[TetR]; C172-LCGGT-RACAG-MXMM" \
  --format json
```

Compile the selected hit list:

```bash
uv run python -m dnadesign.studies.retron_hairpin_design.cli compile \
  --input docs/studies/retron_hairpin_design/msd_design_hit_labels.txt \
  --out-dir /tmp/dnadesign_retron_msd_design_references \
  --format json
```

Materialize single-unit GenBank/PNG outputs after concrete sequence
subcomponents are available:

```bash
uv run python -m dnadesign.studies.retron_hairpin_design.cli materialize \
  --input docs/studies/retron_hairpin_design/msd_design_hit_labels.txt \
  --out-dir /tmp/dnadesign_retron_msd_sequences \
  --payload-sequence TetR=<payload-sequence> \
  --cap-sequence C26=<cap-sequence> \
  --cap-sequence C172=<cap-sequence> \
  --render-format png \
  --format json
```

## Fail-Fast Semantics

- Reject malformed labels.
- Recompute scar-nick profile from left/right bases.
- Reject profile drift.
- Reject non-ligatable `S0`.
- Reject unknown payload/cap registry entries.
- Reject sequence artifact generation when payload or cap sequences are missing;
  route missing cap/shortening inputs to Snapback or missing base-junction inputs
  to scar-nick instead of guessing.
- Reject `--repeat-count`; the compiler emits one MSD unit per design.
- Keep artifact paths and sequence digests nullable until concrete generated
  artifacts are attached.

## Output Posture

Ad hoc outputs belong in explicit transient directories such as
`/tmp/dnadesign_retron_msd_*`. Reader-linked outputs should later be copied
into the owning Reader experiment `inputs/designs/` directory. Do not commit
generated catalogs or visual/sequence artifacts unless the user explicitly asks.
The compiler emits a shallow bundle: top-level `README.md`, `manifest.json`,
`msd_design_catalog_v1.json`, `reference_index.tsv`, and one flat
`references/` directory containing per-design `*.msd_design_reference_v1.json`
files. Do not create one directory per MSD ID for reference-only catalogs, and
use a fresh output directory instead of mixing new output into the legacy
`assets/` layout or stale `references/` files from a different catalog.

The `materialize` route extends the same transient root with
`sequence_manifest.json`, `sequence_index.tsv`, `composition_configs/`, and
`variants/`. Each variant bundle keeps operator-facing files such as
`sequence.gb`, `sequence.fa`, `features.csv`, and `component_span_qa.png`
discoverable at the variant root while retaining service-owned nested visual
contracts for BaseRender. `sequence_index.tsv` carries the `open -R` Finder
reveal command for each GenBank file.

## Service Handoff

- Construct: assemble one selected MSD unit into sequence artifacts.
- Folding: run secondary-structure QA from explicit files or producer bundle.
- BaseRender: render linear/component visual contracts.
- ViennaRNA plotting: fold-layout visualization through Folding, not
  BaseRender.
