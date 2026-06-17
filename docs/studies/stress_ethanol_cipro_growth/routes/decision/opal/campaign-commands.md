## OPAL Campaign Commands

**Last verified:** 2026-06-17

Use this only after `routes/decision/opal/README.md` selects an OPAL campaign
operation.

### Campaign Configs

- Ethanol factor: `src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml`
- Ciprofloxacin factor: `src/dnadesign/opal/campaigns/stress_eth_cip_cipro_rf_sfxi_topn/configs/campaign.yaml`
- AND objective: `src/dnadesign/opal/campaigns/stress_eth_cip_and_rf_sfxi_topn/configs/campaign.yaml`

### Commands

- Catalog route: `uv run ops catalog show opal.downstream.usr-infer-x-active-learning`
- Candidate-table contract audit: `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.candidate_table --config src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/sampling.yaml --validate-existing`
- Candidate provenance audit: `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.provenance --config src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/sampling.yaml`
- Per-ID provenance trace: `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.provenance --config src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/sampling.yaml --id <candidate_id>`
- Campaign config validation, all current objectives:
  - `uv run opal validate -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml --json`
  - `uv run opal validate -c src/dnadesign/opal/campaigns/stress_eth_cip_cipro_rf_sfxi_topn/configs/campaign.yaml --json`
  - `uv run opal validate -c src/dnadesign/opal/campaigns/stress_eth_cip_and_rf_sfxi_topn/configs/campaign.yaml --json`
- Pre-run campaign viewer generation: `uv run opal notebook generate -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml --round latest --force`
- Campaign notebook run: `uv run opal notebook run -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml`
- Label ingest for a measured assay round: `uv run opal ingest-y -c src/dnadesign/opal/campaigns/<campaign_slug>/configs/campaign.yaml --round <observed_round> --in <labels.parquet-or-csv> --unknown-sequences error --apply --json`
- OPAL round run after ingest: `uv run opal run -c src/dnadesign/opal/campaigns/<campaign_slug>/configs/campaign.yaml --round <as_of_round> --json`
- OPAL selection artifact check: `uv run opal verify-outputs -c src/dnadesign/opal/campaigns/<campaign_slug>/configs/campaign.yaml --round <as_of_round> --json`
- OPAL selected-set inspection: `uv run opal selection-set show -c src/dnadesign/opal/campaigns/<campaign_slug>/configs/campaign.yaml --round <as_of_round> --json`
- OPAL selected-set export: `uv run opal selection-set export -c src/dnadesign/opal/campaigns/<campaign_slug>/configs/campaign.yaml --round <as_of_round> --out <selection-set.csv> --format csv --json`
- Post-run status command: `uv run opal status -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml --with-ledger --json`
- Post-run plot command: `uv run opal plot -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml`
- Batch-0 physical synthesis handoff: `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --handoff-id stress-opal-batch0-sfxi-v1 --write --json`
- Measured-round physical synthesis handoff: `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --handoff-id <measured_round_handoff_id> --write --json`

### Lifecycle Harness

The repeated physical-assay loop is:

1. `opal ingest-y` appends measured SFXI labels to the shared observed-label
   sidecar.
2. `opal run` fits/scores/selects for the requested `as_of_round`.
3. `opal verify-outputs` checks the selection artifact against OPAL ledgers.
4. `opal selection-set show` exposes the canonical selected-row contract.
5. Add or update a lifecycle row in
   `docs/studies/stress_ethanol_cipro_growth/record/synthesis_handoffs.yaml`
   with `source_authority=opal_selection_set`, `selection_epoch=opal_model_round`,
   `model_as_of_round=<as_of_round>`, the physical `assay_batch_index`, and
   explicit OPAL `run_id` values.
6. `synthesis_handoff --handoff-id <measured_round_handoff_id>` consumes that
   record-backed selected-set surface and writes campaign-scoped manifest,
   Azenta/GeneWiz workbook, per-sequence GenBank directory, and GenBank
   feature-table artifacts with names prefixed by
   `<handoff_id>__<campaign_slug>__`.

For checked-in lifecycle records, prefer `--handoff-id <handoff_id>` over raw
source flags. Batch zero currently has the checked-in handoff id
`stress-opal-batch0-sfxi-v1`; preview it without `--write` to see exact
campaign-scoped paths, row counts, hashes, and workbook readback status. The
batch-0 selector is the refined BaeR-forward pre-assay plan and requires actual
parsed TFBS regulators, f/e strong sigma-35 slots, d/c exploratory slots, and
16-19 bp spacers.
Measured rounds should follow the same record-driven path after their lifecycle
row exists. Use raw `--source opal-round --round <as_of_round>` only while
drafting a new record or in fixtures.

Do not source measured-round synthesis files from the DenseGen axis probe. The
probe simulates label loops; physical handoff uses real OPAL campaign ledgers.
