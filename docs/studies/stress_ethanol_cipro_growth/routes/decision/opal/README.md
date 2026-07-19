---
doc_id: study-stress-ethanol-cipro-growth-route-decision-opal
surface: study-route-detail
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-07-18
parent_route: ../../README.md
type: route
plane: control-plane
owner_boundary: opal
surface_role: decision
current_state: opal_round0_candidate_review
entry_artifact: usr_prom_eth_cip_opal_candidates
exit_artifact: opal_campaign_records_and_ledgers
---

## OPAL Route Detail

Use this only after `routes/README.md` selects the OPAL campaign surface.

### Surface

- Route state: `opal_round0_candidate_review`
- Entry artifact: `usr_prom_eth_cip_opal_candidates` shared USR candidate table
- Candidate table role: `opal_candidate_feature_table`
- Candidate table X: `latentdna__evo2_7b__context_anchor_mean_bidir_concat`
- Pre-assay seed and candidate provenance: `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/`
- Primary doc: `src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md`

### Detail Surfaces

- Candidate table and label-source semantics: `../../../contexts/opal/candidate-table.md`
- DenseGen TFBS learnability v1 contract/spec: `../../../contexts/opal/densegen-tfbs-learnability-probe-v1.md`
- Response metric metastudy and verified scoring verdict: `../../../contexts/opal/response-metastudy.md`
- MSRB study protocol and interpretation: `../../../contexts/opal/multistate-response-behavior.md`
- Frozen RMF comparator contract: `../../../contexts/opal/response-magnitude-feasibility.md`
- Physical synthesis handoff dev spec: `../../../contexts/opal/synthesis-handoff.md`
- DenseGen synthetic-oracle probe v0: `../../../contexts/opal/densegen-axis-probe-v0.md`
- Manuscript intent and planned response-shape analyses: `../../../contexts/promoter-design-intent.md`
- Campaign configs and commands: `campaign-commands.md`

### Candidate Table Contract

- Dataset id: `usr_prom_eth_cip_opal_candidates`
- Candidate universe: 157160 generated rows plus 25 measured pDual-10 Reader
  rows. Native/reference audit rows stay outside this materialization contract.
- X-selection state: LatentDNA selected this pre-assay X; RegulonDB/native
  appendix visualizations do not gate OPAL campaign readiness.
- SFXI source labels: `_opal/observed_labels.parquet` under the candidate
  table dataset.
- Reader response-window Y source: the verified publication contains 27
  exact labels and eight measured-candidate exclusions.
- SFXI round-0 label input: the three source runs used one deduplicated 35-row
  Reader vec8 pool, independent of the 18-row physical batch-0 synthesis seed.
- SFXI source-run state: each RF/SFXI/top-N run used 35 labels, scored 154785
  candidates, and selected 6 rows. The runs remain metric-review evidence and
  have no executable campaign config.
- MSRB runtime: `secg_msrb_greedy` fits one shared eight-output RF to the
  Reader response-window phenotype. Its ethanol, ciprofloxacin, and AND views
  interpret the same predicted Y through different target masks and rank by
  `behavior_score`. A deterministic allocator requests six slots per view and
  18 unique sequences. Round 0 completed on 2026-07-18 with 27 labels, 154785
  scored candidates, six allocations per view, 18 sequence-unique final
  candidates, and zero output-replay mismatches.
- The campaign is a prospectively frozen greedy learning probe, not evidence
  of RF predictive support or prospective MSRB enrichment.
  `model_support_ready` remains false; selection does not authorize synthesis.
- Candidate eligibility applies OPAL's generic `restriction_site_exclusion`;
  study-specific synthesis constraints remain study-owned.

### Physical Synthesis Handoff

- Lifecycle record: `../../record/synthesis_handoffs.yaml`
- Dev spec: `../../../contexts/opal/synthesis-handoff.md`
- Preview/write: `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --handoff-id <handoff_id> [--write] --json`

### Boundaries

- OPAL reads the study-owned `opal_candidate_feature_table`; do not route this
  as a generic feature matrix.
- OPAL does not own the full DenseGen/Construct/Infer lineage; use
  `decision.opal.batch0.provenance` to verify DenseGen sidecar resolution by
  `id`.
- If pruning shared USR records, use campaign-scoped pruning only. Broad OPAL
  namespace cleanup can delete other campaign columns.
- The unified notebook displays named selection views, campaign records,
  rounds, ledgers, manifest-backed plots, and the logical selection batch.
  Per-record lineage and batch-0 provenance remain study-owned.
- Study visuals enter notebooks only through registered plot APIs and
  `opal.plot_artifact.v1` manifests.
- Response metric review is study-owned and read-only. It uses OPAL's public
  objective facades, verifies the Reader bundle, keeps policy and model gates
  separate, and writes generated workbench evidence. Its `review.py` is not
  campaign state.
- LatentDNA can narrow `X`; OPAL owns label-source validation, training,
  scoring, active selection, and ledgers after labels exist.
- A campaign owns learning; a selection view owns a target; a selection batch
  owns the logical union. Physical synthesis remains study-owned.
- The DenseGen axis probe exercises round mechanics in silico. It is not a
  physical synthesis source and must not fork batch0 or OPAL-ledger semantics.
