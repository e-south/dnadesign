# Retron Hairpin Design Effort

This directory anchors the package-local note for the tracked retron hairpin
design effort. The checked-in record plane lives under
`docs/studies/retron_hairpin_design`, and the current execution surfaces stay in
Cruncher workspaces.

- Study record root: `docs/studies/retron_hairpin_design`
- Primary workspace: `src/dnadesign/cruncher/workspaces/de033`
- Supporting workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`
- MSD design registry: `docs/studies/retron_hairpin_design/msd_design_registry.yaml`
- Study-owned MSD ID compiler:
  `uv run python -m dnadesign.studies.retron_hairpin_design.cli`
- Consolidated context note:
  `src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md`

The MSD ID compiler is intentionally not a top-level project script. It belongs
under the Retron study because it normalizes study-selected construct labels or
typed `retron_msd_compiler_spec_v1` design parts and emits frozen
`msd_design_reference_v1` / `msd_design_catalog_v1` records for downstream
tools such as Reader. Construct and Folding remain workspace-less or
producer-owned task surfaces rather than persistent Retron MSD workspaces.
Solved Snapback and scar-nick primitives enter through public Cruncher APIs, not
study-side imports of Cruncher internals.
