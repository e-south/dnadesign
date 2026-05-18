## OPAL Route Detail

**Last verified:** 2026-05-17

Use this only after `routes/README.md` selects the OPAL campaign surface.

### Surface

- Type: `route-detail`
- Plane: `control-plane`
- Surface role: `decision`
- Owner-boundary: `opal`
- Current state: `candidate_table_materialized_pre_assay`
- Entry artifact: `usr_prom_eth_cip_opal_candidates` shared USR candidate table
- Candidate table role: `opal_candidate_feature_table`
- Candidate table X: `latentdna__evo2_7b__context_anchor_mean_bidir_concat`
- Batch-0 selector: `src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/`
- Candidate provenance audit: `src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/provenance.py`
- Primary doc: `src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md`

### Information Architecture Decision

- Use one shared USR `opal_candidate_feature_table` for the three current
  OPAL campaigns. Its materialized universe is the dense generated promoter
  subset from the LatentDNA view: `source_class=densegen` and `design_family` in
  `background_only`, `ethanol`, `ciprofloxacin`, or `ethanol_ciprofloxacin`.
  Archive SFXI, native, reference, and control rows remain review context.
  The table carries `opal_candidate__source_class`,
  `opal_candidate__design_family`, and `opal_candidate__sfxi_ref__collection_id`
  so the selection filters remain auditable without joining back to LatentDNA.
  DenseGen plan/run/hash provenance is normalized from
  `usr_prom_eth_cip_anchor/_derived/densegen.parquet` during table
  materialization; missing, duplicate, or null sidecar rows are hard failures.
- Keep campaign differences as campaign-scoped OPAL runtime state: objective
  setpoint, model config, `state.json`, notebooks, plots, and `outputs/ledger/`.
- Treat observed SFXI labels as study-level assay truth, not campaign truth; all
  three campaigns train from the same latest observed-label pool after round 0.
- OPAL run/explain and `ingest-y` support shared labels through
  `labels.source.kind: usr_sidecar` at `_opal/observed_labels.parquet`.
- Stress configs set `writeback.prediction_records: ledger_only`, so runs keep
  predictions, scores, selections, and batch ledgers campaign-local instead of
  writing prediction history into the shared `records.parquet`.
- Shared `ingest-y` rejects unknown IDs unless explicitly dropped; it does not
  create candidate rows. Sidecar appends use a local path lock; a records-path
  lock is required before any future shared-record writeback mode.
- Do not mint one USR dataset per campaign unless a campaign needs a distinct
  candidate universe, a distinct `X`, or a distinct data-validation contract.
- LatentDNA owns the evidence for selecting `X`; the study owns materializing
  the table; OPAL owns round state, scoring, active selection, and ledgers.

### Target Label Contract

- Source of truth: an append-only observed-label store keyed by stable candidate
  `id`, observed round or batch, source assay artifact, and SFXI vector schema.
  The current stress path is
  `usr_prom_eth_cip_opal_candidates/_opal/observed_labels.parquet`.
- Consumer contract: every campaign points at the same observed-label source
  and derives its campaign-specific objective through `sfxi_v1` setpoint
  parameters.
- Write contract: campaign runs write predictions, scores, selections, and
  batch ledgers to campaign-local outputs. They should not rewrite the shared
  candidate universe or duplicate observed labels unless explicitly configured.
- Materialization contract: dashboard-friendly summaries may be regenerated
  from the shared label store plus campaign ledgers; they are cache artifacts,
  not primary truth.

### Campaign Configs

- Ethanol factor: `src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml`
- Ciprofloxacin factor: `src/dnadesign/opal/campaigns/stress_eth_cip_cipro_rf_sfxi_topn/configs/campaign.yaml`
- AND objective: `src/dnadesign/opal/campaigns/stress_eth_cip_and_rf_sfxi_topn/configs/campaign.yaml`

### Commands

- Catalog route: `uv run ops catalog show opal.downstream.usr-infer-x-active-learning`
- Candidate-table contract audit: `uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.candidate_table --config src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml`
- Candidate provenance audit: `uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.provenance --config src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml`
- Per-ID provenance trace: `uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.provenance --config src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml --id <candidate_id>`
- Campaign config validation: `uv run opal validate -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml`
- Pre-run campaign viewer generation (writes notebook): `uv run opal notebook generate -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml --round latest --force`
- Campaign notebook run: `uv run opal notebook run -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml`
- Post-run status command (machine-readable): `uv run opal status -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml --with-ledger --json`
- Post-run plot command: `uv run opal plot -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml`

### Boundaries

- OPAL reads the study-owned `opal_candidate_feature_table`; do not route this as a generic feature matrix.
- OPAL does not own the full DenseGen/Construct/Infer lineage; use
  `opal_batch0.provenance` to verify that DenseGen sidecar resolution is
  complete by `id`.
- For transient simulation campaigns that must not mutate shared USR records,
  copy the candidate table into a campaign-local `records.parquet` and use
  `data.location.kind: local`.
- If pruning shared USR records, use campaign-scoped pruning only. Broad OPAL
  namespace cleanup can delete other campaign columns.
- The OPAL notebook is the campaign-specific artifact viewer for records,
  rounds, ledgers, labels, predictions, selected records, and plots.
- LatentDNA can narrow the choice of `X`, but OPAL owns label-source
  validation, training, scoring, active selection, and ledgers after labels
  exist.
