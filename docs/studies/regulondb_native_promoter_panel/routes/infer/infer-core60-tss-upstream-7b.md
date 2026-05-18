---
doc_id: study-regulondb-native-promoter-panel-route-infer-infer-core60-tss-upstream-7b
surface: study-route-detail
study_id: regulondb_native_promoter_panel
owner: dnadesign-maintainers
last_verified: 2026-05-18
parent_route: ../README.md
type: route
plane: control-plane
owner_boundary: infer
surface_role: feature-extraction
current_state: local_complete
entry_artifact: usr_regulondb_native_promoter_core60
exit_artifact: core60_infer_sidecars
---

## Infer Core60 TSS-Upstream 7B Route Detail

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Parent router: [README.md](../README.md).

- Type: `route`
- Plane: `control-plane`
- Surface role: `feature-extraction`
- Owner-boundary: `infer`
- Current state: `local_complete`
- Entry artifact: `usr_regulondb_native_promoter_core60`
- Exit artifact: `_derived/infer` sidecars under `usr_regulondb_native_promoter_core60`
- Config: `src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.core60_tss_upstream.evo2_7b.yaml`
- Runbook: `src/dnadesign/ops/runbooks/presets/infer_regulondb_native_promoter_core60_tss_upstream_7b_batch_with_notify.yaml`
- Route note: This lane extracts derived `analysis_window` views with
  `core60_mean` pooling from the materialized core60 dataset. It
  requests the same intermediate block mean, output-layer mean, mean-per-token
  log likelihood, and total log likelihood sidecars as the native/full lane.
  The current completion inventory reports zero missing vectors/scalars; alias
  rows remain view-level while duplicate physical 60 bp payload rows are reused.
