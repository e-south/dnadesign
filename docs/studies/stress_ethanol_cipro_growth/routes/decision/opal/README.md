---
doc_id: study-stress-ethanol-cipro-growth-route-decision-opal
surface: study-route-detail
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-05-18
parent_route: ../../README.md
type: route
plane: control-plane
owner_boundary: opal
surface_role: decision
current_state: candidate_table_materialized_pre_assay
entry_artifact: usr_prom_eth_cip_opal_candidates
exit_artifact: opal_campaign_records_and_ledgers
---

## OPAL Route Detail

**Last verified:** 2026-05-18

Use this only after `routes/README.md` selects the OPAL campaign surface.

### Surface

- Type: `route`
- Plane: `control-plane`
- Surface role: `decision`
- Owner-boundary: `opal`
- Current state: `candidate_table_materialized_pre_assay`
- Entry artifact: `usr_prom_eth_cip_opal_candidates` shared USR candidate table
- Candidate table role: `opal_candidate_feature_table`
- Candidate table X: `latentdna__evo2_7b__context_anchor_mean_bidir_concat`
- Batch-0 selector: `src/dnadesign/studies/units/stress_ethanol_cipro_growth/opal_batch0/`
- Candidate provenance audit: `src/dnadesign/studies/units/stress_ethanol_cipro_growth/opal_batch0/provenance.py`
- Primary doc: `src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md`

### Detail Surfaces

- Candidate table and label-source semantics:
  `../../../contexts/opal/candidate-table.md`
- Production DenseGen TFBS learnability replacement spec:
  `../../../contexts/opal/densegen-tfbs-learnability-probe-v1.md`
- Historical motif-composition QA precedent:
  `../../../contexts/opal/densegen-motif-qa-k12-s3-v1.md`
- Historical scratch synthetic-oracle probe:
  `../../../contexts/opal/densegen-axis-probe-v0.md`
- Manuscript intent and planned response-shape analyses:
  `../../../contexts/promoter-design-intent.md`
- Campaign configs and commands: `campaign-commands.md`

### Candidate Table Contract

- Dataset id: `usr_prom_eth_cip_opal_candidates`
- Role: `opal_candidate_feature_table`
- Candidate universe: dense generated promoters only; archive SFXI, native,
  reference, and control rows remain review context.
- X column: `latentdna__evo2_7b__context_anchor_mean_bidir_concat`
- X-selection state: LatentDNA selected this pre-assay X; RegulonDB/native
  appendix visualizations do not gate OPAL campaign readiness.
- Shared labels: `_opal/observed_labels.parquet` under the candidate table
  dataset.

### Boundaries

- OPAL reads the study-owned `opal_candidate_feature_table`; do not route this as a generic feature matrix.
- OPAL does not own the full DenseGen/Construct/Infer lineage; use
  `opal_batch0.provenance` to verify that DenseGen sidecar resolution is
  complete by `id`.
- For transient simulation campaigns that must not mutate shared USR records,
  copy the candidate table into an isolated local data root. If the campaign
  uses `labels.source.kind: usr_sidecar`, configure a scratch USR root with
  `data.location.kind: usr`; otherwise use a campaign-local `records.parquet`
  with `data.location.kind: local`.
- If pruning shared USR records, use campaign-scoped pruning only. Broad OPAL
  namespace cleanup can delete other campaign columns.
- The OPAL notebook is the campaign-specific artifact viewer for record
  contract status, rounds, ledgers, label and prediction summaries, selection
  summaries, and plots. Per-record lineage and batch-0 provenance remain in the
  study-owned provenance CLI/API.
- Plot visibility is manifest-backed. Stress bundles and DenseGen scratch
  probes should use configured OPAL plot primitives plus `round_variants` for
  per-round notebook scopes; study-specific visuals only enter OPAL notebooks
  through the registered plot API and `opal.plot_artifact.v1` manifests.
- LatentDNA can narrow the choice of `X`, but OPAL owns label-source
  validation, training, scoring, active selection, and ledgers after labels
  exist.

### Planned Analysis TODOs

- Add a round-aware response-archetype divergence plot after measured
  four-condition labels exist. Compute KL or Jensen-Shannon divergence from the
  underlying response vector `[baseline, ethanol, ciprofloxacin, combined]`,
  not from SFXI alone. Initial visual contract: x-axis `D_KL` to AND-like
  combined-stress target, y-axis `D_KL` to OR-like general-stress target, point
  size SFXI or effect-scaled utility, and point color OPAL round.
- Keep SFXI as an overlay or selection objective: KL maps response shape; SFXI
  ranks whether the response is strong and specific enough to be useful.
- After campaign labels exist, run study-owned mutual-information and feature
  enrichment analyses against DenseGen metadata such as TFBS identity, family,
  count, density, order, spacing, orientation, core promoter variant, and
  distance to promoter elements. Use the same idea in DenseGen probes where
  synthetic labels are available, but keep probe conclusions separate from
  measured promoter-function claims.
- Define behavior classes in response space before architecture clustering.
  UMAP can display architecture clusters, but should not define AND-like or
  OR-like labels.
