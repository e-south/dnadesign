## Retron Compiler Inputs

Convenience label lists and operator-supplied compiler inputs. Persistent
experimental meaning belongs in `../../workbench/design_sets/`.

- `msd_design_hit_labels.txt`: full workbench-backed convenience label list.
- `msd_design_177_194_cap_sources_spec.yaml`: full checked-in materialization
  spec with TetR literal payload and explicit 5'->3' C172/C26 cap/foldback
  segment sequences from `../catalog/msd_cap_sources.yaml`. C26 has no
  subsection topology, so materialization emits the whole `AGGC` segment without
  retained-stem/cap/foldback-return labels.
- `msd_design_177_194_non_ligatable_s0_control_spec.yaml`: non-default
  materialization spec for the operator-requested C172/LCGGG/RACAG/MXMX control.
  It sets `allow_non_ligatable_s0: true`, so the emitted reference marks
  `scar_nick.s0_match_required=false` while profile validation remains strict.
