## stress_ethanol_cipro_growth

- Last verified: 2026-05-02
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Route map: `routes.md`
- Study execution map: `pipeline.yaml`
- LatentDNA binding: `latentdna_binding.yaml`
- Snapshot posture: current
- Preflight posture: available; the remaining main 7B Infer batch is submitted with Notify watchers, and feature-completion advisories remain until those queued GPU jobs finish.

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
- Active study feature-completion target: `evo2_7b`; stale `evo2_20b` row-overlay payloads were removed from the active USR handoffs because that lane is collapsed/debug-required and not part of the current sequence-view completion plan.
- The study phase is `infer_batch_preparation`
- This is a pre-assay representation-triage study. The current notebook/browser surface does not claim a phenotype-validated final `X`.
- Use `uv run ops progress show usr.data-plane.promoter-study-status --json` for the checked-in study record
- Current attention surfaces: queued main 7B fill-infer batch, watcher/GPU job monitoring, and post-run sequence-view completion checks
- Current primary-surface ok: `dataset_overview`, `representation_health_summary`, `design_structure_summary`, `sigma35_ordinal_audit`, `context_robustness_summary`, `candidate_decision_frontier`
- Source-level Sigma-35 inventory is annotation-backed: LatentDNA derives `sig35_variant` from DenseGen plan tokens, DenseGen fixed-element details, USR `seq_annot` `-35` features, or Construct retained-feature bounds. The current merged anchor source resolves every row to a Sigma-35 sequence or b-f ladder value; rows are not hard-omitted from source inventory merely because they are reference, SFXI, or derived core rows.
- Sigma-35 ordinal surfaces use the reverse-alphabetical promoter ladder over the ranked active subset: `f > e > d > c > b` (`a` is not in this study). Annotated unranked hexamers remain visible in inventory and eligible plot/scalar surfaces, but they are excluded from ordinal Spearman rank calculations until an explicit order file ranks them.
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
- Appendix deliverables remain secondary: `appendix_geometry_review`, `appendix_umap_gallery`
- Current appendix attention: none
- Current appendix ok: `appendix_geometry_review`, `appendix_umap_gallery`
- Current regenerated LatentDNA deliverables are ok for `dataset_overview`,
  `representation_health_summary`, `design_structure_summary`,
  `sigma35_ordinal_audit`, `context_robustness_summary`,
  `candidate_decision_frontier`, `appendix_geometry_review`, and
  `appendix_umap_gallery`.
- Current reference plot outputs include the official
  `reference_alignment_summary` appendix plot and an exploratory
  `reference_strength_probe` over the completed reference sidecars.
- The active comparison is 7B construct-insert `seq_mean` anchors versus 7B forward 1 kb context `anchor_mean`. Concat is not part of the current Infer or LatentDNA plan. Forward full-context sequence mean, reverse-complement full-context sequence mean, and reverse-complement context anchor mean remain planned for the main merged handoffs. Reference core60 and reference-context 7B sidecars now exist, but remain diagnostic until a downstream review explicitly promotes them into the decision geometry. Output-layer mean vectors and log-likelihoods are collected diagnostics, not current decision geometry.
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
- `usr_promoter_references` and `usr_sfxi_pdual10_densegen_promoters` also have
  source-local Evo2 7B diagnostic sidecars from the 2026-04-30 dogfood run. Those
  sidecars are useful for source QA, but they do not replace the merged
  `usr_prom_eth_cip_anchor` and `construct_prom_eth_cip_context` downstream
  quota, and they are not a default Notify lane.
- Present matched analysis-core dataset: `construct_prom_eth_cip_reference_core60`
- Present reference context dataset: `construct_prom_eth_cip_reference_contexts`
- Planned aggregate/reference feature dataset: `infer_prom_eth_cip_reference_views_7b`
- Reference Infer is now split into one Notify-compatible lane per USR event
  stream: reference `analysis_window` core60, reference context forward, and
  reference context reverse-complement. The combined reference config remains a
  completion-planning surface only.
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
- The shared `construct_prom_eth_cip_context` handoff also has paired forward and reverse-complement sequence views for every merged anchor row. Sequence-view sidecars are the authoritative orientation/pooling surface for Infer.
- In this record, `anchor_mean` means Infer receives the full emitted 1 kb pDual context, then mean-pools model features over the Construct-provided anchor coordinates. It does not mean Construct or Infer truncates the context to the anchor before model execution. Reverse-complement contexts use the reverse-complement sequence and its emitted-orientation anchor bounds.
- Construct details live in `src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10/runbook.md` and `src/dnadesign/construct/docs/reference/config.md`.
- The downstream reference Infer branch has complete dataset-local sidecars for
  core60 and reference context views. It remains non-blocking for the main study
  state while the study remains in pre-assay representation triage.

### LatentDNA source contract

- Active embedding-bearing LatentDNA sources now read canonical Infer feature sidecars, not USR row-overlay embedding columns.
- `anchor_7b_seq_mean_features` exposes `157164` reusable 7B anchor vectors from `usr_prom_eth_cip_anchor/_derived/infer`.
- `full_context_7b_forward_anchor_mean_features` exposes `157164` reusable 7B forward-context anchor-mean vectors from `construct_prom_eth_cip_context/_derived/infer`.
- LatentDNA also declares canonical Infer scalar-sidecar sources for Evo2
  log-likelihood total and mean-per-token diagnostics. Main merged handoff
  scalar sources are still mostly unfilled; reference core60 and reference
  context scalar sources now resolve locally from generated Infer sidecars.
- Planned sidecar sources for output-layer mean vectors, forward
  full-context sequence mean, reverse-complement full-context sequence mean, and
  reverse-complement context anchor mean still depend on the remaining main
  merged handoff Infer batches. Reference core60 and reference context feature
  sources now resolve locally from generated Infer sidecars.
- The LatentDNA workspace also declares the native reference, reference core60,
  and reference context USR datasets as row-level metadata sources. Anderson and
  W-collection promoter-standard fields remain available as metadata alongside
  the reference feature sidecars.
- Current regenerated LatentDNA source inventory is sidecar/annotation-backed.
  Feature-backed embedding plots still include only rows with existing Infer
  feature aliases; rows that lack vectors are missing feature coverage, not
  ineligible plot categories. Deep validation should be treated as the
  source-of-truth check for stale materialized artifacts after any source or
  sidecar refresh.
- A 2026-04-30 `latentdna inspect source` smoke confirmed all local reference
  feature and scalar sources now resolve at 48 rows per core60 or per
  reference-context geometry. The same day, `latentdna deliverable run
  dataset_overview` and `latentdna deliverable run appendix_geometry_review`
  regenerated the current non-UMAP plot surfaces and deep validation returned
  `status: ok`.
- On 2026-05-02, LatentDNA refreshed the stress `appendix_umap_gallery`
  deliverable after the appendix rename and reference-set summary changes. The
  refreshed workspace snapshot reports all primary and appendix deliverables at
  `status: ok` and `freshness: ok`.
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
  `_derived/infer/feature_vectors.parquet`; USR row-overlay Infer payloads are not a coverage source.
- The sequence-view completion planner reports `missing_products=0`, `314808`
  reusable 7B sequence-view feature vectors, `1258462` missing 7B vectors, `288`
  reusable 7B log-likelihood scalar specs, and `943674` missing 7B
  log-likelihood scalar specs under the full collection plan. The three
  reference lanes are complete; the remaining missing work is in the main merged
  anchor/context lanes.
  Vector counts include both intermediate embeddings and output-layer mean vectors.
  Log-likelihoods are tracked separately through `_derived/infer/feature_scalar_aliases.parquet` and
  `_derived/infer/feature_scalars.parquet`. The reusable main vectors are the canonical sidecar rows
  for `157164` anchor construct-insert intermediate vectors and `157164` forward-context anchor-mean
  intermediate vectors.
- The active handoff datasets no longer carry stale `infer__evo2_20b__*` row-overlay columns or active
  7B row-overlay Infer parts.
- The remaining main Infer work is targeted: `115` anchor intermediate vectors,
  all `157279` anchor output-layer vectors, forward-context sequence-mean
  vectors, the remaining `115` forward-context anchor-mean intermediate vectors,
  all forward-context output-layer vectors, all reverse-complement context
  vectors, and all main log-likelihood scalar sidecars.
- A 12-hour `ops runbook fill-infer --submit` was launched on 2026-04-30 for
  the three remaining main 7B lanes. Each lane has one Notify watcher and one
  RTXP6000 GPU job; the submit gate passed with `missing_products=0`,
  `stale_vectors=0`, and `stale_scalars=0`.
- Partial-fill behavior was rechecked before launch. Completion is feature-spec
  based, not row-skip based: rows with an existing intermediate embedding still
  enqueue missing output-layer vectors and log-likelihood scalar specs. The
  targeted partial-resume regression tests passed.
- `construct_prom_eth_cip_reference_core60` now has complete Infer sidecars for the 48 core views
  (`96` vectors and `96` scalars). `construct_prom_eth_cip_reference_contexts`
  now has complete Infer sidecars for the paired forward and reverse-complement
  context views (`384` vectors and `192` scalar payloads across `seq_mean` and
  `anchor_mean` lanes).
- `usr_promoter_references` and `usr_sfxi_pdual10_densegen_promoters` now have
  source-local diagnostic sidecars (`142` vectors/scalars total across 71 views).
  These are not counted in `ops runbook fill-infer` because the official
  downstream quota is dataset-local completion of the merged anchor/context
  handoffs plus the explicit reference core/context branch.
- Infer should consume explicit sequence views and fail fast on missing required product kinds. It
  should not synthesize missing `analysis_window` or reverse-complement products; Construct owns those
  completion steps.
- `core60_mean`, `seq_mean`, and `anchor_mean` are distinct feature identities.
  Exact repeated input sequences share one Evo2 forward pass through the
  `forward_pass_key`, but they do not share feature-vector keys unless the
  complete feature identity is the same.
- Before submitting a new feature batch, use the sequence-view completion planner to separate
  canonical reusable vectors from missing work:
  `uv run infer validate sequence-view-completion --config <config.yaml> --format json`.
  For host preflight and status loops over large partial datasets, add
  `--mode inventory` so the check counts sidecar aliases and payload-key
  inventory without deriving every missing feature key.
- `usr.data-plane.promoter-study-preflight` now includes `sequence_view_contract` product checks and
  non-blocking `infer_sequence_view_completion` feature-completion checks. Product checks cover the
  merged anchor, merged context, reference core60, and reference context datasets. Feature-completion
  checks run the main 7B sequence-view config and the combined reference 7B planning config without
  loading Evo2; the live reference Notify runbooks use the three split reference lane configs.
  Product checks are current in this checkout, while feature-completion checks still
  report `attention` because output-layer, context sequence-mean,
  reverse-complement, and the remaining `115 + 115` reusable-intermediate gaps
  still need Infer execution in the main merged handoffs.
- `usr.data-plane.promoter-study-status --json` now mirrors that situation at the cheap snapshot
  layer through `sequence_view_contract_state` and `infer_feature_completion_state`. The status route
  is for record-plane situational awareness; use preflight for command-level blockers and host
  execution readiness.
- With the hard-cut generic product-kind vocabulary, generated sidecars that still contain old
  values such as `promoter_insert`, `analysis_core60`, or `context1kb_forward` correctly report
  product-contract `attention`. The local study sidecars have been migrated to
  `construct_insert`, `analysis_window`, `selected_region`, and `realized_context` with recomputed
  sequence-view ids.
- USR dataset parquet files remain generated artifacts ignored by git policy. The local generated
  reference-core/context datasets, updated merged handoff rows, sequence-view sidecars, and
  view-semantics addenda need USR sync or publish handling for another checkout to reproduce them.

### Batch-call readiness and quota deduplication

- As of 2026-04-30, the checked-in ops repertoire for active 7B Infer is six
  sequence-view Notify runbooks: merged anchor construct-insert, merged context
  forward, merged context reverse-complement, reference core60, reference
  context forward, and reference context reverse-complement. The three
  reference lanes completed in real Evo2 7B dogfood runs; `ops runbook
  fill-infer` now skips them and plans only the three incomplete main 7B lanes.
- With `NOTIFY_WEBHOOK_FILE` set to a readable file-backed Slack webhook secret
  and SCC TLS variables set to the system CA bundle, `ops runbook fill-infer`
  renders the three remaining main 7B stress lanes as runnable and skips the
  complete reference lanes. Without that environment, the same command correctly
  blocks Notify-backed batch lanes before submit.
- This workstation has a path-only local env file for that contract at
  `$HOME/.config/dnadesign/notify/env/study_stress_ethanol_cipro.env`; it
  exports `NOTIFY_WEBHOOK_FILE`, `SSL_CERT_FILE`, and `REQUESTS_CA_BUNDLE`
  without storing the webhook URL in the repository.
- Representative no-submit execution gates passed on 2026-04-30 for the merged
  anchor 7B runbook: writable log dirs, live `qsub -verify` for
  `notify-watch.qsub` and `evo2-gpu-infer.qsub`, template QA preflight,
  sequence-view completion validation with zero missing products/stale sidecars,
  `infer run --dry-run`, queue diagnostics, and dry Notify profile smoke.
- Real Evo2 7B dogfood runs passed on 2026-04-30 for the reference core60 lane,
  the paired reference-context lanes, and a source-local reference/SFXI
  diagnostic config. The runs generated 96 vectors/scalars for the 48 core60
  views, 384 vectors plus 192 scalar payloads for the 96 reference context
  records, and 142 vectors/scalars for 71 source-local diagnostic views.
- A fresh public Evo2 adapter recompute over those four jobs on `cuda:0` with
  `bf16` and `batch-size=128` compared 311 contexts, 215 unique forward passes,
  622 vector specs, and 622 scalar specs against persisted sidecars with
  `max_vector_abs_diff=0.0`, `max_scalar_abs_diff=0.0`, and no missing payloads.
  Completed sequence-view reruns now skip Evo2 model loading after resolving
  existing sidecars.
- Live fill-infer submit handles from 2026-04-30:
  anchor construct-insert Notify `4697078`, GPU `4697079`; context forward
  Notify `4697082`, GPU `4697083`; context reverse-complement Notify
  `4697087`, GPU `4697089`. The captured qstat state after submit had all
  three watcher jobs running and all three GPU jobs queued.
- The reference-context data-fidelity audit passed: 48 forward/reverse context
  pairs are exact reverse complements, all reference contexts are 1 kb, all
  reference anchor spans are 60 bp, all emitted reverse anchor bounds map to the
  expected reverse-complement offsets, all reference core60 rows are 60 bp, and
  all SFXI source rows are 60 bp.
- A live Slack webhook smoke sent successfully on 2026-04-30 using the
  file-backed secret-ref route. Dry Notify smoke against existing event logs
  replays historical Infer events unless the watcher cursor is seeded to the
  current `.events.log` byte size; seed to EOF before production `--follow`
  when replay is not intended.
- The 48/48/48 exact-sequence overlap across main and reference Infer lanes is
  expected by design, not a data-fidelity blocker. `construct_prom_eth_cip_reference_core60`
  contributes 48 normalized 60 bp reference anchors into `usr_prom_eth_cip_anchor`;
  the derived reference contexts are likewise present in the shared 1 kb context
  handoff because that handoff is built from the merged anchor set.
- The overlap does not collapse data fidelity because view ids, dataset ids,
  pooling labels, aliases, and sidecar identities remain separate. If the goal
  is dataset-local sidecar completeness, run both main and reference lanes. If
  the goal is strictly minimizing unique Evo2 forward passes, the 144 known
  overlapping inputs can be reused or skipped through an explicit materialization
  policy.

### Next actions

- If you need the current record, refresh the sanctioned snapshot first:
  `uv run ops progress show usr.data-plane.promoter-study-status --json`
- If you need the downstream representation-comparison surface after reading the record-plane snapshot, refresh the LatentDNA workspace snapshot:
  `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json`
- If you need blockers or next-run readiness, switch to `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --command-timeout-seconds 30 --json`
- If you need live batch status, run `qstat -u "$USER"` and inspect the
  corresponding audit JSONs under
  `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/logs/ops/audit/`.
- Treat sidecar-backed 7B intermediate embeddings as the active candidate `X` blocks. Mean-pooled
  output-layer mean vectors and log-likelihoods are collected by Infer for diagnostics/QC, but they are not
  active LatentDNA geometry defaults.
- Do not use UMAP aesthetics, reference-neighbor artifacts, or geodesic pilots as the primary comparison rule
