## regulondb_native_promoter_panel Routes

**Last verified:** 2026-05-18

Use this page as the checked-in one-hop route record for this study. No study-owned OPS status provider is registered for `regulondb_native_promoter_panel`; do not route it through `stress_ethanol_cipro_growth` status surfaces.

- Status: checked-in `../record/status.md`, `../record/datasets.yaml`, `../operations/ops.study.yaml`, and this route map.
- Preflight: owner tool commands in the focused detail pages until this study owns a concrete status/preflight provider.

### Route Index

| Surface | Owner | State | Detail |
| --- | --- | --- | --- |
| Source Intake | `cruncher` | `local_validated` | [Source intake](source/source-intake.md) |
| USR Import | `usr` | `local_validated` | [USR import](source/usr-import.md) |
| Infer Native/Full 7B | `infer` | `local_complete` | [Native/full Infer](infer/infer-native-full-7b.md) |
| Construct Native/Core60/Context | `construct` | `local_validated` | [Core60 Construct](construct/construct-native-core60-context.md) |
| Infer Core60 TSS-Upstream 7B | `infer` | `local_complete` | [Core60 Infer](infer/infer-core60-tss-upstream-7b.md) |
| Fill Remaining Infer | `ops` | `plan_ready` | [Fill remaining Infer](infer/fill-remaining-infer.md) |
| LatentDNA Native Audit | `latentdna` | `local_feature_review_ready` | [LatentDNA native audit](analysis/latentdna-native-audit.md) |
