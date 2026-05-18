## OPAL Route Detail

**Last verified:** 2026-05-18

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

### Detail Surfaces

- Candidate table and label-source semantics:
  `../../../contexts/opal/candidate-table.md`
- Campaign configs and commands: `campaign-commands.md`

### Candidate Table Contract

- Dataset id: `usr_prom_eth_cip_opal_candidates`
- Role: `opal_candidate_feature_table`
- Candidate universe: dense generated promoters only; archive SFXI, native,
  reference, and control rows remain review context.
- X column: `latentdna__evo2_7b__context_anchor_mean_bidir_concat`
- Shared labels: `_opal/observed_labels.parquet` under the candidate table
  dataset.

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
