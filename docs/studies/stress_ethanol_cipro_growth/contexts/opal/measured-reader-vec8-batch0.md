---
id: stress-ethanol-cipro-growth-opal-measured-reader-vec8-batch0
title: Measured reader vec8 batch0 staging
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-08
audience:
  - operator
  - agent
entrypoints:
  code: src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/measured_reader_vec8/
  cli: python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.measured_reader_vec8
---

## Measured Reader Vec8 Batch0 Staging

This surface builds campaign-local round-0 label inputs from measured reader
SFXI `vec8` records. It does not recompute reader math and it does not write
OPAL observed-label sidecars unless the operator separately runs
`opal ingest-y --apply`.

### Inputs

- Reader experiment records under `reader/experiments/2026/*/outputs/manifests/records.json`.
- Reader `sfxi_vec8/vec8` tables named by those records manifests.
- Stress OPAL synthesis manifests under each campaign's
  `outputs/synthesis_handoff/stress-opal-batch0-sfxi-v1/`.
- Current OPAL candidate records:
  `src/dnadesign/usr/datasets/usr_prom_eth_cip_opal_candidates/records.parquet`.
- Reference sequence sources for non-candidate reader aliases:
  `usr_sfxi_pdual10_densegen_promoters`, `usr_promoter_references`, and the
  current stress LatentDNA selected view.

The join is exact. Reader design IDs such as
`pDual-10-SECG-B0-ETH-01` map to synthesis names such as `SECG-B0-ETH-01`.
Historical SFXI and control rows enter campaign CSVs only after their source ID
exists in the OPAL candidate table with sequence and X. The staging command
does not mint candidate rows.

### Outputs

- `src/dnadesign/opal/campaigns/<slug>/inputs/r0/reader_vec8_batch0.csv`
- `src/dnadesign/opal/campaigns/<slug>/inputs/r0/reader_evidence_manifest.json`
- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/measured_reader_vec8/reader_vec8_superset_audit.csv`
- `src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/outputs/measured_reader_vec8/reader_vec8_superset_manifest.json`

The three campaign CSVs are identical shared-label inputs. They contain the
candidate ID, sequence, reader design ID, selected reader experiment, selected
time, and the eight SFXI label components expected by OPAL.

The campaign-local Reader evidence manifests are review inputs for OPAL
notebooks. They carry the same candidate IDs plus Reader provenance fields and
typed artifact paths for the Reader vec8 table, raw kinetics, intensity
overview, and SFXI vec8 heatmap when those artifacts exist. OPAL notebooks read
these manifests for display only; they do not recompute Reader math or mutate
observed labels.

### Current Scope

The current measured reader batch0 set has 35 rows:

- 3 ethanol-responsive candidates from `20260706_sfxi_sensor-panel-m9-glu-secg`.
- 2 ciprofloxacin-responsive candidates from `20260706_sfxi_sensor-panel-m9-glu-secg`.
- 5 AND-responsive candidates from `20260707_sfxi_sensor-panel-m9-glu-secg`.
- 23 historical pDual-10 ES rows measured on plates that used same-plate
  `pDual-10` as the J23105/Anderson YFP anchor.
- 2 measured pDual-10 control promoter rows: `pDual-10-spyp` and
  `pDual-10-sulAp`.

The remaining eight SECG synthesis-manifest candidates have sequence and X in
the candidate table, but they do not yet have reader `vec8` measurements.

`pDual-10` remains the reference anchor and is not emitted as a vec8 label row.
The two control promoter rows are observed-label rows but are excluded from
synthesis-candidate eligibility by exact ID before restriction-site scanning.

### Commands

Preview the staging summary:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.measured_reader_vec8 \
  --repo-root . \
  --reader-root ../reader
```

Write campaign input CSVs and the audit manifest:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.measured_reader_vec8 \
  --repo-root . \
  --reader-root ../reader \
  --write \
  --overwrite
```

Validate one campaign input without mutating labels:

```bash
uv run opal ingest-y \
  -c src/dnadesign/opal/campaigns/secg_ethanol_rf_sfxi_topn/configs/campaign.yaml \
  --round 0 \
  --csv src/dnadesign/opal/campaigns/secg_ethanol_rf_sfxi_topn/inputs/r0/reader_vec8_batch0.csv \
  --unknown-sequences error \
  --json
```

Apply only when the shared observed-label sidecar should be updated. The
current stress configs resolve that sidecar to
`usr_prom_eth_cip_opal_candidates/_opal/observed_labels.parquet`, shared across
the three campaigns. Because the CSVs are identical and the sidecar is shared,
apply one representative campaign input with `--if-exists fail`; do not apply
all three identical files unless using an intentional duplicate policy.
