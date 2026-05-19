## Retron MSD Compiler Inputs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This directory holds study-owned inputs for the narrow Retron MSD compiler.
These are not generated outputs and not experimental rationale.

- `catalog/msd_design_registry.yaml`: payload, cap, and construct normalization
  metadata used by the compiler.
- `catalog/msd_cap_sources.yaml`: concise `C###` cap source lookup with
  explicit 5'->3' sequences.
- `inputs/msd_design_hit_labels.txt`: convenience lab-facing label input mirrored from
  the workbench design-set record.
- `inputs/msd_design_177_194_cap_sources_spec.yaml`: full checked-in
  materialization spec with TetR selected literally and C172/C26 supplied as
  explicit 5'->3' cap/foldback segment sequences.
- `inputs/msd_design_177_194_non_ligatable_s0_control_spec.yaml`:
  operator-requested non-default materialization spec. It explicitly allows
  the C172/LCGGG/RACAG/MXMX control with `s0_match_required=false`; profile
  drift still fails.

Persistent cohort meaning belongs in `../workbench/design_sets/`. Generated
catalogs and sequence bundles belong in explicit transient or caller-owned
output directories.
