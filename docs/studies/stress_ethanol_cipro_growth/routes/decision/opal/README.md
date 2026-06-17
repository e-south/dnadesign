---
doc_id: study-stress-ethanol-cipro-growth-route-decision-opal
surface: study-route-detail
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-06-17
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

**Last verified:** 2026-06-17

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
- Batch-0 selector: `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/`
- Candidate provenance audit: `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/provenance.py`
- Primary doc: `src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md`

### Detail Surfaces

- Candidate table and label-source semantics:
  `../../../contexts/opal/candidate-table.md`
- DenseGen TFBS learnability v1 contract/spec:
  `../../../contexts/opal/densegen-tfbs-learnability-probe-v1.md`
- Physical synthesis handoff dev spec:
  `../../../contexts/opal/synthesis-handoff.md`
- Current DenseGen TFBS implementation/profile surface:
  `../../../../../../src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/README.md`
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

### Physical Synthesis Handoff

- Batch-0 GeneWiz/Azenta files are generated per campaign from the checked-in
  lifecycle record with
  `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --handoff-id stress-opal-batch0-sfxi-v1 --write --json`.
- Batch zero is the BaeR-forward pre-assay seed order: ethanol uses 3 `baeR`,
  1 `cpxR`, 1 `baeR+lexA`, and 1 `cpxR+lexA`; ciprofloxacin uses 4 `lexA`,
  1 `baeR+lexA`, and 1 `cpxR+lexA`; AND uses 4 `baeR+lexA` and 2
  `cpxR+lexA`. The selector requires actual parsed TFBS regulators, f/e
  strong sigma-35 slots, d/c exploratory slots, and 16-19 bp spacers.
- The same command without `--write` is the operator preview: it validates the
  batch-0 selector, checks the lifecycle record, and reports exact
  campaign-local manifest/workbook paths plus current hash/readback status.
- Fetch each workbook from
  `src/dnadesign/opal/campaigns/<campaign_slug>/outputs/synthesis_handoff/stress-opal-batch0-sfxi-v1/azenta_gene_synthesis.xlsx`;
  the matching `synthesis_manifest.csv` in the same folder is the canonical
  ID/provenance/readback artifact.
- After measured labels are ingested and OPAL has run, add a measured-round
  lifecycle row to `../../record/synthesis_handoffs.yaml` with
  `source_authority=opal_selection_set`, `selection_epoch=opal_model_round`,
  `model_as_of_round=<as_of_round>`, `assay_batch_index=<physical_batch_index>`,
  and explicit OPAL `run_id` values. Then generate files with
  `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --handoff-id <measured_round_handoff_id> --write --json`.
- Before generating a measured-round handoff, inspect the OPAL-selected set with
  `uv run opal selection-set show -c src/dnadesign/opal/campaigns/<campaign_slug>/configs/campaign.yaml --round <as_of_round> --json`.
- Measured-round files live under the paths declared by that lifecycle row,
  usually
  `src/dnadesign/opal/campaigns/<campaign_slug>/outputs/synthesis_handoff/stress-opal-r<round>-sfxi-v1/`.
  If a campaign has multiple run IDs for the same round, record the selected
  value as `expected_campaigns[].run_id`; the `--handoff-id` path will pass it
  through to OPAL `selection-set`.
- Handoff lifecycle state is tracked in
  `../../record/synthesis_handoffs.yaml`; generated manifests/workbooks remain
  ignored `outputs/**` artifacts until an operator records hashes and accepts
  the handoff for ordering.

### Boundaries

- OPAL reads the study-owned `opal_candidate_feature_table`; do not route this as a generic feature matrix.
- OPAL does not own the full DenseGen/Construct/Infer lineage; use
  `dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.provenance`
  to verify that DenseGen sidecar resolution is complete by `id`.
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
- Physical synthesis handoff is study-owned. Do not add cloning flanks or
  vendor workbook semantics to OPAL candidate records or generic OPAL core;
  route selected rows through the study synthesis handoff surface.
- The DenseGen axis probe is an in-silico simulation harness. It may exercise
  round mechanics, but it is not a physical synthesis source and must not fork
  batch0 or OPAL-ledger selection semantics.

### Planned Analysis TODOs

- After measured four-condition labels exist, add round-aware KL/Jensen-Shannon
  response-archetype plots over `[baseline, ethanol, ciprofloxacin, combined]`;
  keep SFXI as strength/specificity overlay or selection objective, not the
  response-shape axis.
- Run study-owned mutual-information and enrichment analyses against DenseGen
  TFBS identity, family, count, density, order, spacing, orientation, core
  promoter variant, and distance-to-element metadata.
- DenseGen probe conclusions stay separate from measured promoter-function
  claims; UMAP can show architecture clusters but must not define response labels.
