## stress_ethanol_cipro_growth

- Last verified: 2026-04-28
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Route map: `routes.md`
- Study execution map: `pipeline.yaml`
- LatentDNA binding: `latentdna_binding.yaml`
- Snapshot posture: current
- Preflight posture: available; current blockers are Notify environment/profile setup plus sequence-view feature-completion advisories

### Current datasets

- DenseGen anchor source: `densegen_prom_eth_cip_source` (`present`, shared source)
- SFXI pDual-10 DenseGen source: `usr_sfxi_pdual10_densegen_promoters` (`present`, local source)
- Anchor-only handoff: `usr_prom_eth_cip_anchor` (`present`, shared infer handoff)
- Full-context handoff: `construct_prom_eth_cip_context` (`present`, shared infer handoff)

### Current phase

- Declared phase: `infer_batch_preparation`
- Preferred infer family: `evo2_7b`
- Supported infer families: `evo2_7b`, `evo2_20b` (package-supported; active feature-completion target is 7B)
- LatentDNA browser default family: `evo2_7b`
- Working candidate family: `evo2_7b` full-context anchor-mean intermediate embedding
- Conservative baseline: `evo2_7b` anchor-only intermediate embedding
- Challenger: none currently active; concat is retired from the current Infer and LatentDNA plan
- Secondary/debug-required family: `evo2_20b`
- Active study feature-completion target: `evo2_7b`; historical `evo2_20b` row-overlay payloads were retired from the active USR handoffs because that lane is collapsed/debug-required and not part of the current sequence-view completion plan.
- The study phase is `infer_batch_preparation`
- This is a pre-assay representation-triage study. The current notebook/browser surface does not claim a phenotype-validated final `X`.
- Use `uv run ops progress show usr.data-plane.promoter-study-status --json` for the checked-in study record
- Current attention surfaces: sequence-view feature completion and Notify setup/preflight
- Current primary-surface ok: `dataset_overview`, `representation_health_summary`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`
- Source-level Sigma-35 inventory is annotation-backed: LatentDNA derives `sig35_variant` from DenseGen plan tokens, DenseGen fixed-element details, USR `seq_annot` `-35` features, or Construct retained-feature bounds. The current merged anchor source resolves every row to a Sigma-35 sequence or b-f ladder value; rows are not hard-omitted from source inventory merely because they are reference, SFXI, or derived core rows.
- Sigma-35 ordinal surfaces use the reverse-alphabetical promoter ladder over the ranked active subset: `f > e > d > c > b` (`a` is not in this study). Annotated unranked hexamers remain visible in inventory and compatible plot/scalar surfaces, but they are excluded from ordinal Spearman rank calculations until an explicit order file ranks them.
- Companion visuals: `balanced_design_family_margin_gallery`, `sigma35_margin_ladder_gallery`, `sigma35_stress_margin_gallery`, `context_pair_summary`
- Appendix surfaces remain secondary audit material
- Browser default geometry layout: candidate grid over `intermediate_embedding_7b_anchor_60bp` and `intermediate_embedding_7b_full_context_anchor_mean`

### Current row counts

- DenseGen source row target: `100000`
- DenseGen anchor target before the first full-lane infer gate closes: `100000`
- `densegen_prom_eth_cip_source`: `157160`
- `usr_promoter_references`: `48`
- `construct_prom_eth_cip_reference_core60`: `48`
- `construct_prom_eth_cip_reference_contexts`: `96`
- `usr_sfxi_pdual10_densegen_promoters`: `23`
- `usr_prom_eth_cip_anchor`: `157279`
- `construct_prom_eth_cip_context`: `314558`
- Status JSON route: `evidence.analysis_surfaces.{densegen,latentdna,cluster}`

### Current downstream posture

- LatentDNA: `configured` for downstream comparison; the study-status authority remains the checked-in record plus `usr.data-plane.promoter-study-status`
- LatentDNA gate: `representation_health_summary`
- LatentDNA primary review path: `dataset_overview`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`
- LatentDNA companion visuals: `balanced_design_family_margin_gallery`, `sigma35_margin_ladder_gallery`, `sigma35_stress_margin_gallery`, `context_pair_summary`
- LatentDNA appendix support: `sigma35_centroid_distance_gallery`
- LatentDNA notebook role: plot-first review surface for available 7B sequence-view feature sidecars, with planned reverse-complement/reference and retired diagnostic surfaces kept secondary
- Cluster: `planned`
- OPAL: `not_configured`
- Appendix deliverables remain secondary: `appendix_geometry_audit`, `appendix_umap_gallery`
- Current appendix attention: `appendix_umap_gallery` until the all-row UMAP appendix is regenerated on a machine where that runtime cost is acceptable
- Current appendix ok: `appendix_geometry_audit`
- The active comparison is 7B construct-insert `seq_mean` anchors versus 7B forward 1 kb context `anchor_mean`. Concat is not part of the current Infer or LatentDNA plan. Forward full-context sequence mean, reverse-complement full-context sequence mean, reverse-complement context anchor mean, and reference-view features remain planned until canonical Infer sidecars exist. Mean-pooled output-layer logits and log-likelihoods are collected diagnostics, not current decision geometry.
- Reference alignment remains diagnostic. Native references are biological controls; `analysis_window` reference rows are analysis-only comparability views, not corrected native promoters.

### Reference-view branch

- Present promoter-reference source dataset: `usr_promoter_references`
- Source rows are primer-flank-stripped MG1655 GenBank-projected promoter inserts plus source-backed synthetic promoter standards. J23105 is refreshed from the synthetic GenBank source; full GenBank provenance, projected annotations, strength metadata, derivation intervals, and sequence views are stored in dataset-local sidecars/overlays.
- Present SFXI pDual-10 source dataset: `usr_sfxi_pdual10_densegen_promoters`
- The SFXI source rows are Reader-backed, archive-backed 60 bp pDual-10 DenseGen promoters. They are included in the main merged anchor/context handoff and remain separate from the matched native-reference core60 branch.
- Source-only USR datasets in this checkout now expose generic `source_record`
  sequence views plus mutable view semantics. Those views make dataset identity
  explicit for source, demo, template, SFXI, and DenseGen datasets; they do not
  make those source datasets canonical Infer inputs unless an Infer config
  selects them.
- Present matched analysis-core dataset: `construct_prom_eth_cip_reference_core60`
- Present reference context dataset: `construct_prom_eth_cip_reference_contexts`
- Planned reference feature dataset: `infer_prom_eth_cip_reference_views_7b`
- Sequence-view manifests live as dataset-local `_views/sequence_views.parquet` sidecars rather than a standalone study dataset
- Mutable view-semantics addenda live in dataset-local `_views/view_semantics.parquet` sidecars.
  In this checkout they cover all `629213` active-study source, handoff, and
  reference sequence views, plus all `629753` non-archived local USR sequence
  views. They provide machine-readable `source_family`, `selection_basis`,
  `view_collections`, and `role_tags` without changing stable `view_id`
  identity.
- Native reference rows keep source lengths. Construct derives separate `analysis_window` rows by
  requiring one `sigma70_minus35` and one `sigma70_minus10` annotation, then centering the 60 bp
  analysis window on the midpoint between those sites.
- Over-length references are truncated only in the derived analysis-core dataset. Under-length
  references are expanded only through the pDual-10 replacement template before extracting the 60 bp
  analysis window. The native rows in `usr_promoter_references` are not overwritten.
- Reference core60 rows are merged into `usr_prom_eth_cip_anchor` alongside DenseGen, native references, and SFXI pDual-10 rows.
- In the merged anchor sidecar, every row is exposed as a construct-ready
  `construct_insert` sequence view with `context_kind=anchor_only` and
  `recommended_pooling=seq_mean`. True `analysis_window` product identity
  remains authoritative in `construct_prom_eth_cip_reference_core60`; native or
  designed exact-60 rows are not duplicated as analysis-core products merely
  because their length is 60 bp.
- Reference context rows include paired `realized_context` sequence views with
  `orientation=forward` and `orientation=reverse_complement` plus
  emitted-orientation anchor bounds for `anchor_mean` Infer pooling.
- The shared `construct_prom_eth_cip_context` handoff also has paired forward and reverse-complement sequence views for every merged anchor row. Sequence-view sidecars are the authoritative orientation/pooling surface for Infer; legacy forward rows may retain null `construct__orientation` values in the older Construct overlay.
- In this record, `anchor_mean` means Infer receives the full emitted 1 kb pDual context, then mean-pools model features over the Construct-provided anchor coordinates. It does not mean Construct or Infer truncates the context to the anchor before model execution. Reverse-complement contexts use the reverse-complement sequence and its emitted-orientation anchor bounds.
- Construct details live in `src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10/runbook.md` and `src/dnadesign/construct/docs/reference/config.md`.
- The downstream reference Infer branch remains planned and non-blocking for the main study state while the study remains in pre-assay representation triage

### LatentDNA source contract

- Active embedding-bearing LatentDNA sources now read canonical Infer feature sidecars, not legacy USR row-overlay embedding columns.
- `anchor_7b_seq_mean_features` exposes `157164` reusable 7B anchor vectors from `usr_prom_eth_cip_anchor/_derived/infer`.
- `full_context_7b_forward_anchor_mean_features` exposes `157164` reusable 7B forward-context anchor-mean vectors from `construct_prom_eth_cip_context/_derived/infer`.
- LatentDNA also declares canonical Infer scalar-sidecar sources for Evo2
  log-likelihood total and mean-per-token diagnostics. These sources currently
  expose zero rows because local Infer scalar sidecars have not been generated,
  but the schema is valid and ready for partial fills/resumes.
- Planned sidecar sources for mean-pooled output-layer logits, forward
  full-context sequence mean, reverse-complement full-context sequence mean,
  reverse-complement context anchor mean, reference core60, and reference
  context views currently select zero aliases until Infer fills those vectors.
- The LatentDNA workspace also declares the native reference, reference core60,
  and reference context USR datasets as row-level metadata sources. Anderson and
  W-collection promoter-standard fields remain available as metadata even while
  the reference feature sidecars are still empty.
- Current regenerated LatentDNA source inventory is sidecar/annotation-backed. Feature-backed embedding plots still include only rows with existing Infer feature aliases; SFXI/reference/core rows that lack vectors are missing feature coverage, not ineligible plot categories. Deep validation should be treated as the source-of-truth check for stale materialized artifacts after any source or sidecar refresh.
- Materialized LatentDNA view rows now carry `source_family`,
  `selection_basis`, `view_collections`, `role_tags`, and
  `promoter_standard__*` fields when those fields are present upstream. This
  keeps Anderson and W-collection reference-strength metadata available for plot
  hues after their feature vectors exist.
- `source_class` remains a broad filtering class such as `densegen` or
  `reference_control`; fine-grained provenance is stored separately in
  `source_family`. This prevents context rows whose sequence product is a
  Construct realization from disappearing from DenseGen-filtered context plots.

### Current Infer coverage

- Canonical Infer feature coverage is stored only in `_derived/infer/feature_aliases.parquet` and
  `_derived/infer/feature_vectors.parquet`; active study row-overlay Infer parts have been retired from
  the local generated handoff datasets.
- The sequence-view completion planner reports `missing_products=0`, `314328` reusable main 7B
  sequence-view feature vectors, `1258462` missing main 7B vectors, `0` reusable main 7B
  log-likelihood scalar specs, and `1572790` missing main 7B log-likelihood scalar specs under the
  full collection plan. The reference plan has `480` missing vectors and `480` missing scalar specs.
  Vector counts include both intermediate embeddings and mean-pooled output-layer logits.
  Log-likelihoods are tracked separately through `_derived/infer/feature_scalar_aliases.parquet` and
  `_derived/infer/feature_scalars.parquet`. The reusable main vectors are the canonical sidecar rows
  for `157164` anchor construct-insert intermediate vectors and `157164` forward-context anchor-mean
  intermediate vectors.
- The active handoff datasets no longer carry stale `infer__evo2_20b__*` row-overlay columns or active
  7B row-overlay Infer parts. Historical row-overlay payloads were used only as a migration source,
  then removed after the protected sequence-view sidecars were written.
- The remaining main Infer work is targeted: `115` anchor intermediate vectors, all `157279` anchor
  output-layer vectors, forward-context sequence-mean vectors, the remaining `115` forward-context
  anchor-mean intermediate vectors, all forward-context output-layer vectors, all reverse-complement
  context vectors, and all main log-likelihood scalar sidecars. The reference branch remains `480`
  vectors and `480` scalar specs missing until its 7B sequence-view features are generated. Evo2
  execution is explicitly deferred in this local environment because this device is not the target
  Infer runtime.
- `construct_prom_eth_cip_reference_core60` has no Infer sidecar vectors yet (`96` planned vectors
  across 48 core views), and `construct_prom_eth_cip_reference_contexts` has no Infer sidecar vectors
  yet (`384` planned vectors across 96 context views and two pooling spans).
- Infer should consume explicit sequence views and fail fast on missing required product kinds. It
  should not synthesize missing `analysis_window` or reverse-complement products; Construct owns those
  completion steps.
- Before submitting a new feature batch, use the sequence-view completion planner to separate
  canonical reusable vectors from missing work:
  `uv run infer validate sequence-view-completion --config <config.yaml> --format json`.
- `usr.data-plane.promoter-study-preflight` now includes `sequence_view_contract` product checks and
  non-blocking `infer_sequence_view_completion` feature-completion checks. Product checks cover the
  merged anchor, merged context, reference core60, and reference context datasets. Feature-completion
  checks run the main 7B sequence-view config and the reference 7B sequence-view config without
  loading Evo2; product checks are current in this checkout, while feature-completion checks still
  report `attention` because output-layer, context sequence-mean, reverse-complement, reference, and
  the remaining `115 + 115` reusable-intermediate gaps still need Infer execution.
- `usr.data-plane.promoter-study-status --json` now mirrors that situation at the cheap snapshot
  layer through `sequence_view_contract_state` and `infer_feature_completion_state`. The status route
  is for record-plane situational awareness; use preflight for command-level blockers and host
  execution readiness.
- With the hard-cut generic product-kind vocabulary, generated sidecars that still contain legacy
  values such as `promoter_insert`, `analysis_core60`, or `context1kb_forward` correctly report
  product-contract `attention`. The local study sidecars have been migrated to
  `construct_insert`, `analysis_window`, `selected_region`, and `realized_context` with recomputed
  sequence-view ids.
- USR dataset parquet files remain generated artifacts ignored by git policy. The local generated
  reference-core/context datasets, updated merged handoff rows, sequence-view sidecars, and
  view-semantics addenda need USR sync or publish handling for another checkout to reproduce them.

### Next actions

- If you need the current record, refresh the sanctioned snapshot first:
  `uv run ops progress show usr.data-plane.promoter-study-status --json`
- If you need the downstream representation-comparison surface after reading the record-plane snapshot, refresh the LatentDNA workspace snapshot:
  `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
- If you need blockers or next-run readiness, switch to `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --command-timeout-seconds 30 --json`
- Treat sidecar-backed 7B intermediate embeddings as the active candidate `X` blocks. Mean-pooled
  output-layer logits and log-likelihoods are collected by Infer for diagnostics/QC, but they are not
  active LatentDNA geometry defaults.
- Do not use UMAP aesthetics, reference-neighbor artifacts, or geodesic pilots as the primary comparison rule
