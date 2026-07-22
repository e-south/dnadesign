---
doc_id: study-regulondb-native-promoter-panel-command-groups
surface: study-runtime-command-group-map
study_id: regulondb_native_promoter_panel
owner: dnadesign-maintainers
last_verified: 2026-05-18
entrypoint: self
canonical_payload: pipeline.yaml
---

## RegulonDB Runtime Command Groups

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Use this page when an agent needs command-group orientation without opening the
full `pipeline.yaml`. The full YAML remains the compatibility payload for
checked-in study records and docs-contract checks.

### Lane Index

| Need | Start Here | Canonical Payload |
| --- | --- | --- |
| Cruncher source intake and provenance strata | [Source intake lane](lanes/source-intake.yaml) | `pipeline.yaml:intent` |
| USR dry-run/write import | [USR import lane](lanes/usr-import.yaml) | `pipeline.yaml:command_groups[id=usr_*]` |
| Native source-record to core60 derivation | [Construct lane](lanes/construct.yaml) | `pipeline.yaml:command_groups.construct_core60_tss_upstream` |
| Fill-infer and Evo2 7B sidecar extraction | [Infer lane](lanes/infer.yaml) | `pipeline.yaml:command_groups[id=infer_*]` and `pipeline.yaml:infer` |
| Native/full plus core60 LatentDNA audit | [LatentDNA lane](lanes/latentdna.yaml) | `pipeline.yaml:command_groups.latentdna_native_workspace` |

### Navigation Rule

Open one lane file first, then jump to `pipeline.yaml` only when the task needs
the full machine-readable command list. Keep durable status in
`../../../record/status.md` and owner routing in `../../../routes/README.md`.
