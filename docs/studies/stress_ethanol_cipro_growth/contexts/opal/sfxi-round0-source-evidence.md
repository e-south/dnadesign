---
id: stress-ethanol-cipro-growth-opal-sfxi-round0-source-evidence
title: SFXI round-0 source evidence
owner: dnadesign-maintainers
status: source_evidence
last_verified: 2026-07-13
audience:
  - operator
  - agent
---

## SFXI Round-0 Source Evidence

Three digest-pinned SFXI source runs consumed one deduplicated 35-row Reader
label pool. The pool contains 10 measured pre-assay seed designs, 23 pDual-10
SFXI designs, and 2 pDual-10 control promoters. `pDual-10` is the same-plate
reference and is not a label row.

The 18-row `batch0_synthesis_seed` and the 35-row
`round0_observed_label_pool` are distinct records:

- `batch0_synthesis_seed` records constructs selected before assay data.
- `round0_observed_label_pool` records Reader SFXI vec8 measurements available
  to the three source runs.

Eight seed constructs lack measurements. The remaining 25 label rows come from
pDual-10 SFXI designs and controls, so the observed pool still contains 35
candidate/X-valid labels.

### Source Contract

- Response values use each Reader experiment's nearest snapshot to 12 hours.
- Logic channels use `YFP/CFP`; fluorescence channels use corner-specific
  reference-normalized `YFP/OD600` under the SFXI vec8 contract.
- Candidate identity, sequence parity, and X readiness were checked before
  ingestion.
- The source sidecar is
  `usr_prom_eth_cip_opal_candidates/_opal/observed_labels.parquet`.
- Source run IDs and artifact digests are recorded in
  `docs/studies/stress_ethanol_cipro_growth/record/status.md`.

The campaign-local staging command and the three SFXI executable configs are
absent. These artifacts remain SFXI evidence in their declared y-space; they
are not routes for label promotion or campaign execution.

### RMF Boundary

`secg_rmf_greedy` does not consume SFXI vec8 labels. It requires the typed
eight-component response-window sidecar at
`usr_prom_eth_cip_opal_candidates/_opal/response_window_observed_labels.parquet`.
That sidecar is absent, so RMF ingestion and execution remain inactive.

Reader response-window generation, study-owned metric review, and OPAL commands
are routed through:

- `contexts/opal/response-metastudy.md`
- `contexts/opal/response-magnitude-feasibility.md`
- `routes/decision/opal/campaign-commands.md`
