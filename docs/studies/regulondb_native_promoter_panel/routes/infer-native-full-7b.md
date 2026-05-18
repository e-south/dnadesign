## Infer Native/Full 7B Route Detail

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Parent router: [README.md](README.md).

- Type: `route`
- Plane: `control-plane`
- Surface role: `feature-extraction`
- Owner-boundary: `infer`
- Current state: `local_complete`
- Entry artifact: `usr_regulondb_native_promoters`
- Exit artifact: `_derived/infer` sidecars under `usr_regulondb_native_promoters`
- Config: `src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full.evo2_7b.yaml`
- Runbook: `src/dnadesign/ops/runbooks/presets/infer_regulondb_native_promoter_native_full_7b_batch_with_notify.yaml`
- Route note: This lane extracts native `source_record` views with
  `seq_mean` pooling and requests the intermediate block mean, output-layer
  mean, mean-per-token log likelihood, and total log likelihood sidecars.
  Local preflight validates the config, resolves the Notify event path, and the
  current completion inventory reports zero missing vectors/scalars.
