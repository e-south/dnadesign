## stress_ethanol_cipro_growth

- Last verified: 2026-04-25
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Route map: `routes.md`
- Study execution map: `pipeline.yaml`
- LatentDNA binding: `latentdna_binding.yaml`
- Snapshot posture: current
- Preflight posture: not requested

### Current datasets

- DenseGen anchor source: `densegen_prom_eth_cip_source` (`present`, shared source)
- Anchor-only handoff: `usr_prom_eth_cip_anchor` (`present`, shared infer handoff)
- Full-context handoff: `construct_prom_eth_cip_context` (`present`, shared infer handoff)

### Current phase

- Declared phase: `infer_batch_preparation`
- Preferred infer family: `evo2_7b`
- Supported infer families: `evo2_7b`, `evo2_20b`
- LatentDNA browser default family: `evo2_7b`
- Working candidate family: `evo2_7b` full-context anchor-mean intermediate embedding
- Conservative baseline: `evo2_7b` anchor-only intermediate embedding
- Challenger: `evo2_7b` anchor-plus-anchor-mean concat
- Secondary/debug-required family: `evo2_20b`
- The study phase is `infer_batch_preparation`
- This is a pre-assay representation-triage study. The current notebook/browser surface does not claim a phenotype-validated final `X`.
- Use `uv run ops progress show usr.data-plane.promoter-study-status --json` for the checked-in study record
- Current attention surfaces: none
- Current primary-surface ok: `dataset_overview`, `representation_health_summary`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`
- Sigma-35 ordinal surfaces use the reverse-alphabetical promoter ladder over the active subset: `f > e > d > c > b` (`a` is not in this study)
- Companion visuals: `balanced_design_family_margin_gallery`, `sigma35_margin_ladder_gallery`, `sigma35_stress_margin_gallery`, `context_pair_summary`
- Appendix surfaces remain secondary audit material
- Browser default geometries: `intermediate_embedding_7b_anchor_60bp`, `pooled_logits_7b_anchor_60bp`, `intermediate_embedding_7b_full_context_1kb`, `pooled_logits_7b_full_context_1kb`, `intermediate_embedding_7b_full_context_anchor_mean`, `intermediate_embedding_7b_anchor_plus_full_context_concat`, `intermediate_embedding_7b_anchor_plus_anchor_mean_concat`

### Current row counts

- DenseGen source row target: `100000`
- DenseGen anchor target before the first full-lane infer gate closes: `100000`
- `densegen_prom_eth_cip_source`: `157160`
- `usr_prom_eth_cip_anchor`: `157164`
- `construct_prom_eth_cip_context`: `157164`
- Status JSON route: `evidence.analysis_surfaces.{densegen,latentdna,cluster}`

### Current downstream posture

- LatentDNA: `configured` for downstream comparison; the study-status authority remains the checked-in record plus `usr.data-plane.promoter-study-status`
- LatentDNA gate: `representation_health_summary`
- LatentDNA primary review path: `dataset_overview`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`
- LatentDNA companion visuals: `balanced_design_family_margin_gallery`, `sigma35_margin_ladder_gallery`, `sigma35_stress_margin_gallery`, `context_pair_summary`
- LatentDNA appendix support: `sigma35_centroid_distance_gallery`
- LatentDNA notebook role: plot-first review surface for the seven-geometry 7B-first pre-assay ladder, with appendix and debug material kept secondary
- Cluster: `planned`
- OPAL: `not_configured`
- Appendix deliverables remain secondary: `appendix_geometry_audit`, `appendix_umap_gallery`
- Current appendix attention: none
- Current appendix ok: `appendix_geometry_audit`, `appendix_umap_gallery`
- The active comparison is `anchor_60bp` versus `full_context_anchor_mean`, with `full_context_1kb` retained as an orientation/appendix view and pooled-logit surfaces treated as diagnostics rather than the default decision rule
- Reference alignment remains diagnostic. Native references are biological controls; any future `analysis_core60` reference rows are analysis-only comparability views, not corrected native promoters.

### Reference-view branch

- Present promoter-reference source dataset: `usr_promoter_references`
- Source rows are primer-flank-stripped MG1655 GenBank-projected promoter inserts plus source-backed synthetic promoter standards. J23105 is refreshed from the synthetic GenBank source; full GenBank provenance, projected annotations, strength metadata, derivation intervals, and sequence views are stored in dataset-local sidecars/overlays.
- Planned matched analysis-core dataset: `construct_prom_eth_cip_reference_core60`
- Planned reference context dataset: `construct_prom_eth_cip_reference_contexts`
- Planned reference feature dataset: `infer_prom_eth_cip_reference_views_7b`
- Sequence-view manifests live as dataset-local `_views/sequence_views.parquet` sidecars rather than a standalone study dataset
- The downstream core/context/infer branch remains planned and non-blocking for the main study state while the study remains in pre-assay representation triage

### Next actions

- If you need the current record, refresh the sanctioned snapshot first:
  `uv run ops progress show usr.data-plane.promoter-study-status --json`
- If you need the downstream representation-comparison surface after reading the record-plane snapshot, refresh the LatentDNA workspace snapshot:
  `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
- If you need blockers or next-run readiness, switch to `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
- Treat `intermediate_embedding_*` and `pooled_logits_*` as candidate `X` blocks; use `log_likelihood_per_token_*` only as scalar side channels
- Do not use UMAP aesthetics, reference-neighbor artifacts, or geodesic pilots as the primary comparison rule
