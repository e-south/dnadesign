---
id: stress-ethanol-cipro-growth-opal-sfxi-round0-source-evidence
title: SFXI round-0 source evidence
owner: dnadesign-maintainers
status: source_evidence
last_verified: 2026-07-18
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

- Source-run directories are stored under
  `src/dnadesign/studies/units/stress_ethanol_cipro_growth/workbench/source_evidence/opal_sfxi_round0/`.
  Their slugs and run IDs are immutable provenance; they are not executable
  OPAL campaigns.
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

### Response-Window and MSRB Boundary

`secg_msrb_greedy` does not consume SFXI vec8 labels. Its model target is the
typed eight-component Reader response-window Y stored at
`usr_prom_eth_cip_opal_candidates/_opal/response_window_labels_v5/observed_labels.parquet`.
The verified sidecar contains 27 exact labels and remains separate from this
SFXI source evidence. MSRB is applied only after the shared model predicts that
Y. The completed RMF round is retained separately as frozen comparator
evidence; no SFXI vec8 row entered either response-window label table.

Reader response-window generation, study-owned metric review, and OPAL commands
are routed through:

- `contexts/opal/response-metastudy.md`
- `contexts/opal/multistate-response-behavior.md`
- `contexts/opal/response-magnitude-feasibility.md` (frozen RMF comparator)
- `routes/decision/opal/campaign-commands.md`
