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
  explicit 5'->3' cap/foldback segment sequences. The pES-retron-177 entry is
  the operator-specified C172/LCGGG/RACAG/MXMX control and sets
  `allow_non_ligatable_s0: true`; profile drift still fails.

Typed specs may also carry manual custom payload/cap IDs when the same spec
provides literal `payload_sequences` and `cap_sequences`. Payload primitive
sources are not accepted until a dedicated public payload-source contract
exists.

Persistent cohort meaning belongs in `../workbench/design_sets/`. Generated
catalogs and sequence bundles belong in explicit transient or caller-owned
output directories.
