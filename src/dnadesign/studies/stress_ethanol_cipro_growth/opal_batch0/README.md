# OPAL Batch 0 Handoff

This directory owns the pre-assay seed selection for the three stress /
ethanol / ciprofloxacin OPAL campaigns.

Batch 0 is not OPAL model selection. It is a reviewed handoff that combines
DenseGen design metadata, the current LatentDNA representation choice, and the
campaign setup rules. After measured labels exist, ingest them with
`opal ingest-y --observed-round 0`, then run OPAL with `opal run --labels-as-of
0`.

## Candidate Feature Table

OPAL reads the USR dataset
`src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates/records.parquet`.
That dataset is a candidate feature table, not just a matrix. Each promoter row
must carry stable identity, sequence, audit/provenance metadata, and one
fixed-length vector-valued X column:

`latentdna__evo2_7b__context_anchor_mean_bidir_concat`

The X value is the fixed-length Fwd+RC 1 kb context-anchor Evo2 7B intermediate
embedding vector. It is not a UMAP coordinate, a centroid distance, an assay
label, or a phenotype claim.

## Selection

Preview the batch-0 review rows without writing generated outputs:

```bash
uv run python -m dnadesign.studies.stress_ethanol_cipro_growth.opal_batch0.select \
  --config src/dnadesign/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml
```

Write the configured generated review tables only after the operator wants
local artifacts:

```bash
uv run python -m dnadesign.studies.stress_ethanol_cipro_growth.opal_batch0.select \
  --config src/dnadesign/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml \
  --write
```

The review table is study-generated until labels are measured. Do not treat
LatentDNA margins as phenotype evidence, and do not promote UMAP placement or
centroid-nearest rank into the selection rule.
