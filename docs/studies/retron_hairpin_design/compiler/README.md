## Retron MSD Compiler Inputs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This directory holds study-owned inputs for the narrow Retron MSD compiler.
These are not generated outputs and not experimental rationale.

- `msd_design_registry.yaml`: payload, cap, and construct normalization
  metadata used by the compiler.
- `msd_design_hit_labels.txt`: convenience lab-facing label input mirrored from
  the workbench design-set record.

Persistent cohort meaning belongs in `../workbench/design_sets/`. Generated
catalogs and sequence bundles belong in explicit transient or caller-owned
output directories.
