# Retron Hairpin Design Effort

This directory anchors the package-local note for the tracked retron hairpin
design effort. The checked-in record plane lives under
`docs/studies/retron_hairpin_design`, and the current execution surfaces stay in
Cruncher workspaces.

- Study record root: `docs/studies/retron_hairpin_design`
- Primary workspace: `src/dnadesign/cruncher/workspaces/de033`
- Supporting workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`
- MSD compiler inputs: `docs/studies/retron_hairpin_design/compiler/`
- Study-owned MSD ID compiler:
  `uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app`
- Consolidated context note:
  `src/dnadesign/cruncher/docs/dev/audits/2026-04-19-retron-p4-hairpin-variant.md`

Open source modules by responsibility:

- `interfaces/cli/`: Typer commands, input parsing, output rendering, and
  operator next-step messages.
- `compiler/references.py`: MSD label-to-reference compilation.
- `compiler/catalog_bundle.py`: catalog/reference bundle writing.
- `compiler/materialization.py`: Construct-backed single-unit sequence bundle
  materialization.
- `compiler/exceptions.py`: fail-fast compiler exception type.
- `catalog/`: label parsing, typed `retron_msd_compiler_spec_v1` parsing,
  primitive selectors, and registry loading.
- `outputs/`: single-unit sequence payloads, output-layout constants,
  stale-artifact guards, manifests, GenBank/plot publication, and rasterization.
- `status/`: study status and preflight provider implementation.

The MSD ID compiler is intentionally not a top-level project script. It belongs
under the Retron study because it normalizes study-selected construct labels or
typed `retron_msd_compiler_spec_v1` design parts and emits frozen
`msd_design_reference_v1` / `msd_design_catalog_v1` records for downstream
tools such as Reader. Construct and Folding remain workspace-less or
producer-owned task surfaces rather than persistent Retron MSD workspaces.
Solved Snapback and scar-nick primitives enter through public Cruncher APIs, not
study-side imports of Cruncher internals.
