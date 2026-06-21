## Retron Compiler Inputs

Convenience label lists and operator-supplied compiler inputs. Persistent
experimental meaning belongs in `../../workbench/design_sets/`.

- `msd_design_hit_labels.txt`: full workbench-backed convenience label list.
- `msd_design_177_194_cap_sources_spec.yaml`: full checked-in materialization
  spec with TetR literal payload and explicit 5'->3' C172/C26 cap/foldback
  segment sequences from `../catalog/msd_cap_sources.yaml`. The first label is
  the operator-specified C172/LCGGG/RACAG/MXMX control, so this spec sets
  `allow_non_ligatable_s0: true` and the emitted reference marks
  `scar_nick.s0_match_required=false`. C26 has no subsection topology, so
  materialization emits the whole `AGGC` segment without
  retained-stem/cap/foldback-return labels.
- `teto_pwm_trim_rescue_v1.spec.yaml`: nine-design bidirectional TetR PWM trim rescue spec
  for retron26 control, retron43 target scaffold, and one selected
  DE033-compatible target context. Payload entries are literal 5'->3'
  sequences from the Cruncher monotypic TetR elite with trim metadata; durable experimental meaning stays in
  `../../workbench/design_sets/teto_pwm_trim_rescue_v1.yaml`.
