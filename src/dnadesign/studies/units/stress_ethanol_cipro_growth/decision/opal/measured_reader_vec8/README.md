---
id: stress-ethanol-cipro-growth-measured-reader-vec8
title: Measured reader vec8 staging
owner: stress_ethanol_cipro_growth
status: active
---

# Measured Reader Vec8 Staging

This package stages measured reader SFXI `vec8` rows as round-0 OPAL label
inputs for the three SECG campaigns.

It has three responsibilities:

- Load latest reader `sfxi_vec8/vec8` records from experiment manifests.
- Resolve exact reader design IDs to stress OPAL candidate IDs, sequences, and
  available X values.
- Write identical campaign-local `inputs/r0/reader_vec8_batch0.csv` files plus
  campaign-local `inputs/r0/reader_evidence_manifest.json` files and a global
  audit table.

It does not import reader internals, recompute SFXI math, or apply OPAL labels.
Applying labels remains an explicit `opal ingest-y --apply` operator action
because the current campaigns share the USR observed-label sidecar.

The current batch0 CSVs contain 35 rows: 10 measured SECG synthesis-manifest
rows, 23 historical pDual-10 ES rows, and the `pDual-10-spyp` /
`pDual-10-sulAp` measured controls. `pDual-10` is the same-plate reference
anchor and is not emitted as a label row.

The evidence manifests are for notebook review. They include candidate ID,
Reader design ID, selected Reader experiment, selected time, and typed paths to
Reader-produced plots or records when available. They are not an alternate label
source.

Run the CLI from the dnadesign repository root:

```bash
uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.measured_reader_vec8 \
  --repo-root . \
  --reader-root ../reader \
  --write \
  --overwrite
```
