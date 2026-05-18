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
That dataset is one shared candidate feature table for the three current OPAL
campaign configs, not one dataset per campaign and not just a matrix. It is the
dense generated promoter subset from the current LatentDNA view: rows whose
`source_class` is `densegen` and whose `design_family` is one of
`background_only`, `ethanol`, `ciprofloxacin`, or `ethanol_ciprofloxacin`.
Reference, native-audit, archive SFXI, and control rows stay in LatentDNA
review surfaces but do not enter the first OPAL candidate universe. Each
promoter row must carry stable identity, sequence, audit/provenance metadata,
and one fixed-length vector-valued X column:

`latentdna__evo2_7b__context_anchor_mean_bidir_concat`

The X value is the fixed-length block-normalized Fwd+RC 1 kb context-anchor
Evo2 7B intermediate embedding vector. It is not a raw Infer vector concat, a
UMAP coordinate, a centroid distance, an assay label, or a phenotype claim.
The generated table also carries `opal_candidate__source_class`,
`opal_candidate__design_family`, and `opal_candidate__sfxi_ref__collection_id`
so the dense generated subset remains auditable without a LatentDNA join.

Campaign-specific predictions, selections, and run artifacts use OPAL-owned
campaign-local `outputs/ledger/` state. Observed SFXI labels are shared assay
truth: every ethanol, ciprofloxacin, and AND campaign should train on the same
latest observed-label pool while keeping its own setpoint and objective state.
The current shared-label source is the USR sidecar
`usr_prom_eth_cip_opal_candidates/_opal/observed_labels.parquet`.
Legacy OPAL training labels are campaign-slug scoped; the stress campaign
configs now use `labels.source.kind: usr_sidecar` for `ingest-y`, run/explain
training, and candidate exclusion, plus `writeback.prediction_records:
ledger_only` so run predictions do not mutate the shared `records.parquet`.
Fork the USR candidate table only if a future campaign uses a different
candidate universe or a different `X` contract.
For this study the OPAL-writeback surface is the shared observed-label sidecar
plus campaign-local ledgers; `records.parquet` remains the candidate/X table.
Shared sidecar appends use a local path lock during load/merge/write.
For scratch simulations that must leave the shared USR table untouched, copy
the candidate table and configure the campaign with `data.location.kind:
local`. A records-path lock is required before enabling any future shared-record
prediction writeback mode.

Dry-run the candidate-table materialization contract before writing generated
USR data:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.candidate_table \
  --config src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml
```

Add `--write` only when the operator is ready to create or replace the generated
`records.parquet`.

Check the ID-level provenance chain without reading the vector-valued X payload:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.provenance \
  --config src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml
```

Trace one candidate through DenseGen, anchor records, Construct views, Infer
aliases, LatentDNA rows, and OPAL records:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.provenance \
  --config src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml \
  --id d74fb7d33565954da98f9f3f9f3fa7016e03799d
```

Some DenseGen plan/run/hash fields are incomplete in row-level source records.
The candidate-table materializer resolves those fields by `id` from
`usr_prom_eth_cip_anchor/_derived/densegen.parquet` and fails if the sidecar is
missing, duplicated, or null for a selected candidate. Do not infer those values
from OPAL campaign ledgers or from a campaign-local label history column.

## Selection

Preview the batch-0 review rows without writing generated outputs:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.select \
  --config src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml
```

Write the configured generated review tables only after the operator wants
local artifacts:

```bash
uv run python -m dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_batch0.select \
  --config src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_batch0/sampling.yaml \
  --write
```

The review table is study-generated until labels are measured. Do not treat
LatentDNA margins as phenotype evidence, and do not promote UMAP placement or
centroid-nearest rank into the selection rule.
