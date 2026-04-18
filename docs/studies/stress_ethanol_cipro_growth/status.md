## stress_ethanol_cipro_growth

- Last verified: 2026-04-17
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Route map: `routes.md`
- Study execution map: `pipeline.yaml`
- LatentDNA binding: `latentdna_binding.yaml`
- Snapshot posture: current
- Preflight posture: not requested

### Current datasets

- DenseGen anchor source: `densegen/study_stress_ethanol_cipro` (`present`, shared source)
- Anchor-only handoff: `promoter/stress_ethanol_cipro_anchor_set` (`present`, shared infer handoff)
- Full-context handoff: `promoter/stress_ethanol_cipro_construct_contexts` (`present`, shared infer handoff)

### Current phase

- Declared phase: `infer_batch_preparation`
- Preferred infer family: `evo2_20b`
- Supported infer families: `evo2_20b`, `evo2_7b`
- The study is still in infer preparation; LatentDNA is a downstream comparison surface, not the record-plane authority for current study status
- The sanctioned current-record surface is `uv run ops progress show usr.data-plane.promoter-study-status --json`
- Current attention surfaces: `dataset_overview`, `reference_margin_analysis`, `representation_comparison`
- Context and health diagnostics remain in the record, and appendix surfaces remain secondary audit material

### Current row counts

- DenseGen source row target: `100000`
- DenseGen anchor target before the first full-lane infer gate closes: `100000`
- `densegen/study_stress_ethanol_cipro`: `157160`
- `promoter/stress_ethanol_cipro_anchor_set`: `157164`
- `promoter/stress_ethanol_cipro_construct_contexts`: `157164`
- Status JSON route: `evidence.analysis_surfaces.{densegen,latentdna,cluster}`

### Current downstream posture

- LatentDNA: `configured` for downstream comparison; the study-status authority remains the checked-in record plus `usr.data-plane.promoter-study-status`
- LatentDNA primary attention surfaces: `dataset_overview`, `reference_margin_analysis`, `representation_comparison`
- LatentDNA notebook role: plot-first review surface with `Geometry audit` and `Comparison audit` as secondary tabs
- Cluster: `planned`
- OPAL: `not_configured`
- Appendix surfaces remain secondary: `reference_margin_gallery_synthetic_centroids`, `appendix_umap_gallery`
- The active comparison is `anchor_60bp` versus `full_context_1kb`

### Next actions

- If you need the current record, refresh the sanctioned snapshot first:
  `uv run ops progress show usr.data-plane.promoter-study-status --json`
- If you need the downstream representation-comparison surface after reading the record-plane snapshot, refresh the LatentDNA workspace snapshot:
  `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
- If you need blockers or next-run readiness, switch to `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
- Treat `intermediate_embedding_*` and `pooled_logits_*` as candidate `X` blocks; use `log_likelihood_per_token_*` only as scalar side channels
- Do not use UMAP aesthetics, cluster correspondence, or model-agreement surfaces as the primary comparison rule
