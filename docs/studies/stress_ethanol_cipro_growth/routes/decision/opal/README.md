---
doc_id: study-stress-ethanol-cipro-growth-route-decision-opal
surface: study-route-detail
study_id: stress_ethanol_cipro_growth
owner: dnadesign-maintainers
last_verified: 2026-07-15
parent_route: ../../README.md
type: route
plane: control-plane
owner_boundary: opal
surface_role: decision
current_state: round0_selection_review
entry_artifact: usr_prom_eth_cip_opal_candidates
exit_artifact: opal_campaign_records_and_ledgers
---

## OPAL Route Detail

Use this only after `routes/README.md` selects the OPAL campaign surface.
### Surface

- Route state: `round0_selection_review`
- Entry artifact: `usr_prom_eth_cip_opal_candidates` shared USR candidate table
- Candidate table role: `opal_candidate_feature_table`
- Candidate table X: `latentdna__evo2_7b__context_anchor_mean_bidir_concat`
- Batch-0 selector: `src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/`
- Primary doc: `src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md`

### Detail Surfaces

- Candidate table and label-source semantics: `../../../contexts/opal/candidate-table.md`
- DenseGen TFBS learnability v1 contract/spec: `../../../contexts/opal/densegen-tfbs-learnability-probe-v1.md`
- Response metric metastudy and verified scoring verdict: `../../../contexts/opal/response-metastudy.md`
- RMF contract and promotion gate: `../../../contexts/opal/response-magnitude-feasibility.md`
- Physical synthesis handoff dev spec: `../../../contexts/opal/synthesis-handoff.md`
- DenseGen synthetic-oracle probe v0: `../../../contexts/opal/densegen-axis-probe-v0.md`
- Manuscript intent and planned response-shape analyses: `../../../contexts/promoter-design-intent.md`
- Campaign configs and commands: `campaign-commands.md`

### Candidate Table Contract

- Dataset id: `usr_prom_eth_cip_opal_candidates`
- Role: `opal_candidate_feature_table`
- Candidate universe: 157160 generated rows plus 25 measured pDual-10 Reader
  rows. Native/reference audit rows stay outside this materialization contract.
- X column: `latentdna__evo2_7b__context_anchor_mean_bidir_concat`
- X-selection state: LatentDNA selected this pre-assay X; RegulonDB/native
  appendix visualizations do not gate OPAL campaign readiness.
- SFXI source labels: `_opal/observed_labels.parquet` under the candidate
  table dataset.
- Prospective response-window labels: the verified v1 publication contains 27
  exact labels and eight measured-candidate exclusions.
- SFXI round-0 label input: the three source runs used one deduplicated 35-row
  Reader vec8 pool, independent of the 18-row physical batch-0 synthesis seed.
- SFXI source-run state: each RF/SFXI/top-N run used 35 labels, scored 154785
  candidates, and selected 6 rows. The runs remain metric-review evidence and
  have no executable campaign config.
- RMF runtime: `secg_rmf_greedy` owns one shared eight-output RF and ethanol,
  ciprofloxacin, and AND selection views. Round 0 used the promoted Y and a
  deterministic allocator to produce six slots per view and 18 unique sequences.
- The completed run is a prospectively frozen learning probe, not evidence that
  the RF is a supported phenotype predictor. `model_support_ready` remains
  false, and no synthesis is authorized.
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
- OPAL notebooks display campaign records, rounds, ledgers, and manifest-backed
  plots. The unified notebook displays named selection views and the logical
  selection batch. Per-record lineage and batch-0 provenance remain study-owned.
- Study visuals enter notebooks only through registered plot APIs and
  `opal.plot_artifact.v1` manifests.
- Response metric review is study-owned and read-only. It calls OPAL's public
  objective facades, verifies Reader's public response-window bundle, keeps
  policy and model gates separate, and writes only generated workbench evidence.
  The generated metastudy `review.py` is an evidence viewer, not campaign state.
- LatentDNA can narrow `X`; OPAL owns label-source validation, training,
  scoring, active selection, and ledgers after labels exist.
- A campaign owns learning; a selection view owns a target; a selection batch
  owns the logical union. Physical synthesis remains study-owned.
- The DenseGen axis probe is an in-silico simulation harness. It may exercise
  round mechanics, but it is not a physical synthesis source and must not fork
  batch0 or OPAL-ledger selection semantics.
