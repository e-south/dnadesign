---
doc_id: study-regulondb-native-promoter-panel-route-construct-native-core60-context
surface: study-route-detail
study_id: regulondb_native_promoter_panel
owner: dnadesign-maintainers
last_verified: 2026-05-18
parent_route: ../README.md
type: route
plane: data-plane
owner_boundary: construct
surface_role: derivation
current_state: local_validated
entry_artifact: usr_regulondb_native_promoters
exit_artifact: usr_regulondb_native_promoter_core60
---

## Construct Native/Core60/Context Route Detail

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Parent router: [README.md](../README.md).

- Type: `route`
- Plane: `data-plane`
- Surface role: `derivation`
- Owner-boundary: `construct`
- Current state: `local_validated`
- Entry artifact: `usr_regulondb_native_promoters`
- Exit artifact: `usr_regulondb_native_promoter_core60`
- Workspace: `src/dnadesign/construct/workspaces/study_regulondb_native_promoter_panel`
- Route note: Native `source_record` rows remain source views. The checked-in
  core60 route emits a new `analysis_window` dataset by taking `[0,60)` from
  the native 81 bp source window, using the declared TSS offset `60`. This is
  not -10/-35 box centering. The 2026-04-29 materialization wrote 3,182
  sequence-view rows and 3,181 canonical 60 bp sequence rows; the row-count
  difference is expected USR sequence deduplication for duplicate derived
  windows.
