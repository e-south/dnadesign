---
doc_id: study-retron-hairpin-design-command-groups
surface: study-runtime-command-group-map
study_id: retron_hairpin_design
owner: dnadesign-maintainers
last_verified: 2026-05-18
entrypoint: docs/studies/retron_hairpin_design/operations/runtime/command-groups/README.md
canonical_payload: pipeline.yaml
---

## Retron Runtime Command Groups

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Use this page when an agent needs command-group orientation without opening the
full `pipeline.yaml`. The full YAML remains the compatibility payload for OPS
status/preflight and docs-contract checks.

### Lane Index

| Need | Start Here | Canonical Payload |
| --- | --- | --- |
| Compile MSD labels or specs into frozen references | [Compiler lane](lanes/compiler.yaml) | `pipeline.yaml:command_groups[id=msd_design_reference_catalog]` |
| Emit GenBank, native-structure PNG, and review plots for selected sequence sources or explicit S0 controls | [Materialize lane](lanes/materialize.yaml) | `pipeline.yaml:command_groups[id=msd_single_unit_materialize]` |
| Solve or audit released-product cap geometry | [Snapback lane](lanes/snapback.yaml) | `pipeline.yaml:command_groups[id=snapback_released_*]` |
| Regenerate scar-nick base-junction panel outputs | [Scar-nick lane](lanes/scar-nick.yaml) | `pipeline.yaml:command_groups[id=scar_nick_profile_panel]` |
| Contrast mismatch-boundary language only | [YIU lane](lanes/yiu.yaml) | `pipeline.yaml:command_groups[id=yiu_boundary_check]` |

### Navigation Rule

Open one lane file first, then jump to `pipeline.yaml` only when the task needs
the full machine-readable command list. Keep task routing in
`../../../routes/README.md`, persistent cohort meaning in `../../../workbench/`,
and current-state facts in `../../../record/status.md`.
