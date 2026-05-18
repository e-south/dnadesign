## Retron Design Evidence

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This file holds durable evidence pointers for the Retron hairpin design status
note. Keep first-hop routing in `../../routes/README.md` and current state in
`../status.md`.

### Routing And Workbench

- Study route map: `docs/studies/retron_hairpin_design/routes/README.md`.
- Study workbench: `docs/studies/retron_hairpin_design/workbench/README.md`.
- Direction vocabulary: `docs/studies/retron_hairpin_design/workbench/ontology/directions.yaml`.
- Authoritative design set:
  `docs/studies/retron_hairpin_design/workbench/design_sets/scar_nick_profile_panel_v1.yaml`.

### Regenerable Primitive Evidence

- Released-product solve bundle:
  `src/dnadesign/cruncher/workspaces/de033/outputs/released_solve`. Generated
  outputs are ignored and may be absent after workspace cleanup.
- MSD-HOPV5 visual comparison:
  `src/dnadesign/cruncher/workspaces/msd-HOPV5_snapback`.
- Scar-nick profile-panel bundles:
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_bbsI_hf`
  and
  `src/dnadesign/cruncher/workspaces/scar_nick_teto/outputs/scar_nick/teto_upstream_processing_paqci_core_panel`.
  Current BbsI-HF plus PaqCI coverage reaches 13/14 active profile buckets,
  with `WMWM` still uncovered under the strict catalog policy.

### Compiler And Composition Evidence

- Study command ladder:
  `docs/studies/retron_hairpin_design/operations/runtime/command-groups/pipeline.yaml`.
- Scar-nick base-junction context:
  `docs/studies/retron_hairpin_design/contexts/cruncher/scar-nick-base-junction.md`.
- Linear ssDNA composition handoff:
  `docs/studies/retron_hairpin_design/contexts/composition/linear-ssdna-composition.md`.
- Study-owned MSD design registry:
  `docs/studies/retron_hairpin_design/compiler/catalog/msd_design_registry.yaml`.
- Study-owned cap source lookup:
  `docs/studies/retron_hairpin_design/compiler/catalog/msd_cap_sources.yaml`.
- Study-selected MSD label list:
  `docs/studies/retron_hairpin_design/compiler/inputs/msd_design_hit_labels.txt`.
- Generic linear ssDNA composition dev spec:
  `docs/dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md`.
- Generic linear ssDNA composition execution plan:
  `docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md`.

Study-owned MSD design-reference compilation is available through
`uv run python -m dnadesign.studies.studies.retron_hairpin_design.interfaces.cli.app`. It
consumes user-provided labels plus study registry metadata, emits a shallow
design-reference bundle with `README.md`, `manifest.json`,
`reference_index.tsv`, `msd_design_catalog_v1.json`, and flat per-design
`msd_design_reference_v1` records under `references/` into an explicit
caller-chosen transient directory, and is intentionally not a top-level
`retron-msd` script or persistent workspace family.

### Tool References

- Released-product workflow:
  `src/dnadesign/cruncher/docs/guides/snapback_released_workflow.md`.
- Released-product artifact reference:
  `src/dnadesign/cruncher/docs/reference/released_snapback_artifacts.md`.
- YIU workflow: `src/dnadesign/cruncher/docs/guides/yiu_workflow.md`.
- Consolidated retron/P4 and YIU note:
  `src/dnadesign/cruncher/docs/dev/audits/2026-04-19-retron-p4-hairpin-variant.md`.
