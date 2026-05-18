## LatentDNA Native Audit Route Detail

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Parent router: [README.md](../README.md).

- Type: `route`
- Plane: `data-plane`
- Surface role: `downstream-analysis`
- Owner-boundary: `latentdna`
- Current state: `local_feature_review_ready`
- Entry artifact: native/full and later core60 7B vector and scalar feature surfaces
- Workspace: `src/dnadesign/latentdna/workspaces/regulondb_native_promoter_panel`
- Binding: `docs/studies/regulondb_native_promoter_panel/contexts/latentdna/binding.yaml`
- Route note: Native cohorts use `regulondb__*` fields. They must not derive
  DenseGen metadata or alias native sigma factors into `sig35_variant`.
  The native/full and core60 contracts both name intermediate embeddings,
  output-layer means, and log-likelihood scalar diagnostics from Infer sidecars.
  Native `seq_mean` and core60 mean are sequence-position means over
  causal/prefix-conditioned Evo2 token states in the emitted forward
  orientation: native rows pool the 81 bp source-record window, while core60
  rows pool the derived 60 bp TSS-upstream analysis window.
  The current local snapshot is feature-backed and reports the primary
  decision deliverables as current; future missing sidecars must be expressed
  through explicit planned source roles, not hidden fallback behavior.
