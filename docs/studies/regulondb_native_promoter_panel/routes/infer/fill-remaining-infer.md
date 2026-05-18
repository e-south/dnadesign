---
doc_id: study-regulondb-native-promoter-panel-route-infer-fill-remaining-infer
surface: study-route-detail
study_id: regulondb_native_promoter_panel
owner: dnadesign-maintainers
last_verified: 2026-05-18
parent_route: ../README.md
type: route
plane: control-plane
owner_boundary: ops
surface_role: batch-ergonomics
current_state: plan_ready
entry_artifact: study_infer_runbooks
exit_artifact: fill_infer_plan_and_audit_json
---

## Fill Remaining Infer Route Detail

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Parent router: [README.md](../README.md).

- Type: `route`
- Plane: `control-plane`
- Surface role: `batch-ergonomics`
- Owner-boundary: `ops`
- Current state: `plan_ready`
- Entry artifact: checked-in study `execution_surfaces` or explicit Infer runbook paths
- Exit artifact: one fill plan plus one workspace-scoped audit JSON per executed runnable lane
- Command: `uv run ops runbook fill-infer --study-dir docs/studies/regulondb_native_promoter_panel`
- Submit command: `uv run ops runbook fill-infer --study-dir docs/studies/regulondb_native_promoter_panel --submit`
- Route note: Ops discovers the study's Infer runbooks, runs the sequence-view
  completion inventory, skips complete lanes, blocks lanes with missing
  sequence products or stale sidecars, and plans only lanes with missing
  vectors/scalars. The primitive is study-record based and can also accept
  repeated `--runbook` paths, so it is not tied to RegulonDB or promoter
  semantics.
