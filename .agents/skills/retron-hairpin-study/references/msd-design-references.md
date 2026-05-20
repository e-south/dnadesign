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
If the user provides a typed `retron_msd_compiler_spec_v1`, parse labels or
explicit design parts from the spec. If any primitive is missing, route instead
of guessing.

User-provided labels are authoritative for a live request. Do not switch to a
checked-in cohort spec, alter left/right bases, or choose a scar-compatible
analog just to make validation pass. If validation needs an intentional
non-ligatable-S0 control, require `--allow-non-ligatable-s0` or
`allow_non_ligatable_s0: true` and still preserve the exact label.

Minimum complete materialized deliverable:

- complete reference fields above
- concrete literal payload sequence for every payload id
- concrete cap sequence for every cap id, either literal or from an explicit
  public primitive source

User language such as "outputs", "deliverables", "exports", "GenBank",
"plots", or "open in Finder" means the materialized deliverable, not just the
catalog JSON bundle. If those sequence subcomponents are unavailable, stop on
the missing IDs or primitive route instead of substituting a compile-only
catalog.

## Boundary

- Study-owned package: `dnadesign.studies.studies.retron_hairpin_design`.
- Registry: `docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml`.
- Authoritative design set:
  `docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml`.
- Convenience label input: `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt`.
- Cap source lookup:
  `docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources.yaml`.
- Full cohort materialization spec:
  `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml`.
- Compiler spec: `retron_msd_compiler_spec_v1` with `labels`, `designs`,
  `payload_sequences`, and `cap_sequences`.
- Do not expose a top-level `retron-msd` script.
- Do not create Construct or Folding workspaces per ID.
- Snapback and scar-nick solve missing primitive parts; the compiler validates
  selected parts into reference contracts and one MSD unit sequence per design.
- Workspace-derived primitives must be loaded through
  `dnadesign.cruncher.snapback` or `dnadesign.cruncher.scar_nick` public APIs,
  not by importing `dnadesign.cruncher.src.*` or scraping internals in study
  code.
- A materialized MSD unit is 5' flank + left base, payload primary, a
  user-selected cap/foldback segment, payload complement, right base + 3'
  flank. Snapback subsection labels are optional topology metadata, not a
  requirement for literal sequence assembly.
- Manual custom payload or cap IDs can compile from a typed spec when that spec
  supplies literal `payload_sequences` and `cap_sequences`. Payload primitive
  sources are not accepted until a dedicated public payload-source contract
  exists.
- `C###` cap IDs are symbolic. Known cap sequences live in
  `msd_cap_sources.yaml`: `C26=AGGC`, `C43=tCCTCAGcccGCTGAGGa`, and the
  selected `C172-C176` de033 source labels. The compiler does not infer de033
  sequence or topology from an id pattern.
- Do not add a repeat count or chain complete MSD units together.
- Reader consumes `msd_design_catalog_v1`, not live dnadesign internals.

## Commands

Lint one label:

```bash
uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app lint \
  --id "pES-retron-177-msd[TetR]; C172-LCGGG-RACAG-MXMX" \
  --allow-non-ligatable-s0 \
  --format json
```

Lint a typed compiler spec:

```bash
uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app lint \
  --spec path/to/retron_msd_compiler_spec.yaml \
  --format json
```

Compile the selected hit list:

```bash
uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app compile \
  --input docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt \
  --allow-non-ligatable-s0 \
  --out-dir /tmp/dnadesign_retron_msd_design_references \
  --format json
```

Materialize the full checked-in cohort from explicit 5'->3' cap/foldback
segments:

```bash
uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app materialize \
  --spec docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml \
  --out-dir /tmp/dnadesign_retron_msd_sequences \
  --render-format png \
  --format json
```

The same `compile` and `materialize` commands accept `--spec` instead of
`--id` or `--input`; do not mix those input surfaces in one command.
`C###` cap IDs are not implicit de033 lookups; materialization specs must
provide each cap as a literal 5'->3' sequence or an explicit public primitive
source. Payload sequences are literal today. If topology is absent, the whole
supplied cap/foldback segment is materialized as one segment and subsection
labels are omitted.

For persistent experimental provenance, update the workbench design set and run
records. Generated compile/materialize bundles remain transient unless an
owning Reader experiment snapshots them.

When the user asked for deliverables or Finder output, prefer `materialize`
directly. A successful materialize response includes `finder_open` and writes
`manifest/indexes/sequence_index.tsv`; use that index to reveal per-design
GenBank files or verify plot paths.

## Fail-Fast Semantics

- Reject malformed labels.
- Recompute scar-nick profile from left/right bases.
- Reject profile drift.
- Reject non-ligatable `S0`.
- Admit non-ligatable `S0` only under explicit control opt-in; never substitute
  bases or profiles to satisfy the default rule.
- Reject unknown payload/cap registry entries unless the typed spec supplies
  explicit literal sequences for those IDs.
- Reject mixed `--spec` and ad hoc `--id`/`--input` sources.
- Reject primitive source selectors that return multiple options; use
  `selector.mode=rank` for the explicit combination until a separate expansion
  contract exists.
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
use a fresh output directory instead of mixing new output into the legacy `assets/`
layout or stale `references/` files from a different catalog.

The `materialize` route uses a headful bundle layout that keeps the top level
limited to `README.md`, `manifest/`, and `variants/`. Machine-readable root
metadata is grouped by ontology: bundle manifests under `manifest/bundle/`,
catalogs and frozen references under `manifest/catalog/`, indexes under
`manifest/indexes/`, and generated composition configs under
`manifest/configs/composition/`. Each
`variants/<construct_id>__<msd_design_id>/` directory is grouped into
`sequences/` for forward and reverse-complement GenBank/FASTA plus feature CSV,
`plots/` for `secondary_structure.native.png`, `composition_overview.svg`, and
`composition_overview.png`,
`manifest/` for curated per-variant metadata grouped into `composition/`,
`construct/`, `folding/`, `provenance/`, `reviews/`, and `visual/`, and
`runtime/construct/` for the producer bundle. `runtime/construct/manifest/`
mirrors the same semantic grouping for producer metadata. Variant directory
names include the lab construct id/variant number and preserve scar-nick
ontology in the suffix, for example
`pES-retron-177__msd-tetr-C172-LCGGG-RACAG-MXMX`, with cap, left base, right
base, and mismatch profile uppercase. `manifest/indexes/sequence_index.tsv`
carries the `open -R` Finder reveal command for each forward GenBank file.

For a materialize request, verify the bundle by checking the expected variant
count and at least these per-design artifacts:

- `sequences/forward.gb`
- `sequences/reverse_complement.gb`
- `plots/secondary_structure.native.png`
- `plots/composition_overview.svg`
- `plots/composition_overview.png`

`secondary_structure.native.png` must be rasterized from the ViennaRNA native
secondary-structure SVG after folding status `ok`. `composition_overview.svg`
must be the two-row review with secondary structure first and the BaseRender
component span second; `composition_overview.png` must be its high-resolution
raster sibling for review workflows. Legacy `component_span.png`,
`secondary_structure.png`, `secondary_structure.svg`, and
`component_span_and_folding.png` files are stale/wrong curated deliverables.

## Service Handoff

- Construct: assemble one selected MSD unit into sequence artifacts.
- Folding: run secondary-structure QA from explicit files or producer bundle.
- BaseRender: render linear/component visual contracts.
- ViennaRNA plotting: fold-layout visualization through Folding, not
  BaseRender.
