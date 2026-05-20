## OPAL Candidate Table Context

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-17

Use one shared USR `opal_candidate_feature_table` for the three current OPAL
campaigns. Its materialized universe is the dense generated promoter subset
from the LatentDNA view: `source_class=densegen` and `design_family` in
`background_only`, `ethanol`, `ciprofloxacin`, or `ethanol_ciprofloxacin`.
Archive SFXI, native, reference, and control rows remain review context.

The table carries `opal_candidate__source_class`,
`opal_candidate__design_family`, and `opal_candidate__sfxi_ref__collection_id`
so selection filters remain auditable without joining back to LatentDNA.
DenseGen plan/run/hash provenance is normalized from
`usr_prom_eth_cip_anchor/_derived/densegen.parquet` during table
materialization; missing, duplicate, or null sidecar rows are hard failures.

### Campaign Boundary

- Keep campaign differences as campaign-scoped OPAL runtime state: objective
  setpoint, model config, `state.json`, notebooks, plots, and `outputs/ledger/`.
- Treat observed SFXI labels as study-level assay truth, not campaign truth;
  all three campaigns train from the same latest observed-label pool after
  round 0.
- OPAL run/explain and `ingest-y` support shared labels through
  `labels.source.kind: usr_sidecar` at `_opal/observed_labels.parquet`.
- Stress configs set `writeback.prediction_records: ledger_only`, so runs keep
  predictions, scores, selections, and batch ledgers campaign-local instead of
  writing prediction history into the shared `records.parquet`.
- Shared `ingest-y` rejects unknown IDs unless explicitly dropped; it does not
  create candidate rows.
- Do not mint one USR dataset per campaign unless a campaign needs a distinct
  candidate universe, a distinct `X`, or a distinct data-validation contract.
- LatentDNA owns the completed evidence for selecting `X`; the study owns
  materializing the table; OPAL owns round state, scoring, active selection,
  and ledgers. RegulonDB/native appendix views are review context, not OPAL
  readiness inputs.

### Target Label Contract

- Source of truth: an append-only observed-label store keyed by stable
  candidate `id`, observed round or batch, source assay artifact, and SFXI
  vector schema. The current stress path is
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
