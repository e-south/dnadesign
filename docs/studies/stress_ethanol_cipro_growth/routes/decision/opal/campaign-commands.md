## OPAL Campaign Commands

**Last verified:** 2026-05-19

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
- Post-run status command: `uv run opal status -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml --with-ledger --json`
- Post-run plot command: `uv run opal plot -c src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml`
