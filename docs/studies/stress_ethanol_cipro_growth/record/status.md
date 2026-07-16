## stress_ethanol_cipro_growth

- Last verified: 2026-07-15
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Route map: `../routes/README.md`
- Study execution map: `../operations/runtime/command-groups/pipeline.yaml`
- LatentDNA binding: `../contexts/latentdna/binding.yaml`
- USR root: `src/dnadesign/usr/datasets`
- Snapshot posture: synchronized after local pull from BU SCC `cluster`
- Preflight posture: available; supported Evo2 7B Infer sequence-view lanes are complete. The next main-path readiness question is the OPAL candidate feature table and pre-assay campaign handoff, not GPU Infer submission or RegulonDB-native appendix visualization.

### Datasets

- DenseGen anchor source: `densegen_prom_eth_cip_source` (`present`, 157160 rows)
- Native/reference promoter source: `usr_promoter_references` (`present`, 48 rows)
- SFXI pDual-10 DenseGen source: `usr_sfxi_pdual10_densegen_promoters` (`present`, 23 rows)
- Construct template seed: `usr_pdual10_plasmid_template` (`present`, 1 row)
- Anchor-only handoff: `usr_prom_eth_cip_anchor` (`present`, 160460 rows)
- Full-context handoff: `construct_prom_eth_cip_context` (`present`, 320920 rows)
- Reference core60 handoff: `construct_prom_eth_cip_reference_core60` (`present`, 48 rows)
- Reference context handoff: `construct_prom_eth_cip_reference_contexts` (`present`, 96 rows)
- RegulonDB native promoter source: `usr_regulondb_native_promoters` (`present`, 3182 rows; regulatory-interaction sidecar populated with 3426 rows; BioCyc GO sidecars populated for 203/205 interacting regulators)
- RegulonDB native core60 source: `usr_regulondb_native_promoter_core60` (`present`, 3181 sequence rows; 3180 unambiguous parent-resolved rows feed the TF-axis audit after one duplicate core60 sequence collapses two NhaR-only native parents)
- OPAL candidate feature table: `usr_prom_eth_cip_opal_candidates`
  (`present`, 157185 rows; role `opal_candidate_feature_table`; dense
  generated promoter subset plus measured pDual-10 SFXI/control round-0 rows
  from the broader 160460-row LatentDNA view; 157183 DenseGen-backed rows have
  renderable TFBS metadata and 2 pDual-10 controls are explicit exemptions)
- Logical reference feature entry: `infer_prom_eth_cip_reference_views_7b` (`planned`, not separately materialized; feature payloads live in dataset-local `_derived/infer/` sidecars)

### Study Phase

- Declared phase: `opal_candidate_table_pre_assay`; the nested OPAL
  decision route is in `round0_selection_review`. The study remains pre-assay
  because no revised selection has been authorized for synthesis.
- LatentDNA has selected the pre-assay X. RegulonDB/native appendix visualization
  does not gate OPAL readiness.
- DenseGen growth: `parallel_optional`
- Merged anchor set: `complete`
- Construct context expansion: `complete`
- Evo2 7B sequence-view Infer sidecars: `complete`
- Preferred infer family: `evo2_7b`
- Supported infer families: `evo2_7b`, `evo2_20b`
- Secondary/debug-required family: `evo2_20b`
- LatentDNA browser default family: `evo2_7b`
- Next surface: `docs/studies/stress_ethanol_cipro_growth/routes/decision/opal/README.md`
- Selected pre-assay `X`: `intermediate_embedding_7b_context_anchor_mean_bidir_concat`
- Conservative DenseGen-plan baseline: `intermediate_embedding_7b_anchor_60bp`
- Strength-standard interpretation lens: `intermediate_embedding_7b_full_context_anchor_mean`

Pre-assay representation triage has selected the OPAL `X`. It is not a
phenotype-validated final representation, but it is the chosen fixed-length
input for the next OPAL active-learning handoff.

### Infer Coverage

- Sequence-view product contracts: `4/4 ok`
- Infer feature-completion checks: `2/2 ok`
- Required 7B feature views: `802540`
- Reusable 7B vectors: `1605080`
- Reusable 7B scalars: `963048`
- Missing products: `0`
- Missing vectors: `0`
- Missing scalars: `0`
- Stale vectors: `0`
- Stale scalars: `0`

Completed 7B sidecar lanes:

- `anchor_construct_insert_seq_mean_7b`
- `context_forward_seq_and_anchor_mean_7b`
- `context_reverse_complement_seq_and_anchor_mean_7b`
- `reference_analysis_window_core60_mean_7b`
- `reference_context_forward_seq_and_anchor_mean_7b`
- `reference_context_reverse_complement_seq_and_anchor_mean_7b`

`fill-infer --no-submit` skips the six supported complete lanes and the three
unsupported lanes. It plans no runnable GPU jobs for the declared row quota.

### Downstream Posture

- DenseGen analysis surface: `attention`; the source dataset is ready, but the operator-visible plot inventory contains stale artifacts and should be refreshed before using the plots as evidence.
- LatentDNA X-selection: `complete`; `intermediate_embedding_7b_context_anchor_mean_bidir_concat` is the selected pre-assay X for the OPAL candidate table.
- LatentDNA native TF-axis overlay: `verified`; the deliverable renders over the existing context-anchor bidirectional view after RegulonDB native core60 rows were appended through `usr_prom_eth_cip_anchor` and `construct_prom_eth_cip_context`, regulatory interactions were populated, and matching 7B feature sidecars were filled.
- RegulonDB functional annotation sidecars: `verified`; `usr_regulondb_native_promoters/_relations/` now carries BioCyc KB 29.6 regulator GO terms, promoter-regulator-GO term rows, and regulator coverage rows. LatentDNA has a separate BioCyc GO biological-process appendix plot that reuses the native plan-margin tail tables for interpretation. These are source-backed annotation sidecars, not OPAL inputs or mechanistic labels.
- Cluster: `planned`; use `../routes/README.md` for the exploratory-clustering handoff.
- OPAL: `round0_selection_review`; three digest-pinned SFXI source runs remain
  immutable diagnostic evidence. Their run IDs are
  `r0-2026-07-09T18:37:10+00:00` (ethanol),
  `r0-2026-07-09T18:37:49+00:00` (ciprofloxacin), and
  `r0-2026-07-09T18:38:31+00:00` (AND). Each used 35 labels, scored 154785
  candidates, and selected 6 rows. No executable campaign configs exist for
  these source runs.
- RMF campaign: `secg_rmf_greedy` is the sole executable stress-study OPAL
  config. It declares one shared eight-output RF with ethanol,
  ciprofloxacin, and AND selection views. Round 0 completed as run
  `r0-2026-07-16T01:32:16+00:00` from 27 exact promoted response-window labels.
  Each view received six sequence-unique slots through the declared
  round-robin next-best-unallocated allocator. The preferred lists contained
  one cross-view overlap; the AND view advanced once, producing the required
  18-sequence batch. This is a prospectively frozen learning probe.
  `model_support_ready` remains false, and the completed selection does not
  authorize synthesis.
- Candidate TFBS metadata: `verified`; the 2026-07-12 rematerialization repaired
  79505 dropped `densegen__used_tfbs_detail` values by binding the authoritative
  DenseGen sidecar. The readiness contract now fails if any DenseGen-backed row
  lacks BaseRender-compatible TFBS detail or required-regulator metadata.
- SFXI round-0 source label pool: the SFXI source runs used one
  deduplicated 35-row vec8 sidecar with 10 measured synthesis-manifest rows, 23
  pDual-10 SFXI rows, and 2 pDual-10 control rows. No campaign-local staging
  command exists. The physical 18-row batch-0
  synthesis seed remains provenance for the pre-assay order, not a row-count
  constraint on the SFXI source label pool.
- Response metric metastudy: `verified`. Canonical beta=1 gamma=1 SFXI recomputes
  exactly, but 18 top-six slots collapse to 11 sequences, 2 candidates occur in
  all three campaigns, and scoring is effect dominated. Reader now publishes a
  verified `reader.response_window.bundle.v5` for 8 experiments, 7 reductions,
  413 design/reduction rows, 206500 joint bootstrap rows, and 12 repeated design
  IDs. The primary reduction is the 4-8 hour post-event log mean; the other
  declared windows, normalized linear AUC, and delta remain response sensitivity
  analyses. The strongest descriptive fixed challenger is PLS4 over the primary
  eight-component summary, with weakest selection-view response-separation and
  feasibility Spearman values of 0.45. Retrospective grouped enrichment is strongest for
  ciprofloxacin and weakest for ethanol, but all exact 95% intervals include
  0.5 and do not establish calibrated success probabilities. Under the
  study's time and assay-capacity constraints, the prospective policy assigns
  six sequence-unique slots per selection view. The study observation contract
  selects the latest reviewed source where source agreement is accepted,
  excludes unresolved source disagreements, and uses the selected Reader
  source's joint bootstrap without an equal-experiment or population-level
  uncertainty claim. Eight repeated candidates have selected sources; ES22,
  ES25, ES28, and ES30 are excluded. The verified observation bundle and typed
  OPAL publication contain 27 exact labels and eight measured-candidate
  exclusions. The exact configured campaign RF is evaluated
  separately from descriptive challengers and is the only basis for the model
  support gate. Generic `opal ingest-y` cannot publish this source. This is not
  a synthesis handoff.
- Synthesis handoff: `generated_pending_acceptance` scaffold exists in
  `synthesis_handoffs.yaml`; batch zero is the granular single-axis/AND
  pre-assay seed order with actual parsed TFBS regulator and slot-pattern
  checks. Preview it with
  `uv run python -m dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff --handoff-id stress-opal-batch0-sfxi-v1 --json`.
  Batch-0 generated files are study-owned source-evidence artifacts. The
  lifecycle record pins manifest, workbook, GenBank-directory, and feature-table
  hashes; workbook and GenBank readback must pass before ordering.

LatentDNA decision surfaces: see
`../contexts/latentdna/review-surfaces.md` for the review-surface inventory,
browser-control semantics, UMAP caveats, and pooling guardrails.

### Next Actions

- Keep the SFXI round-0 selections in their declared y-space as metric-review
  evidence, not a synthesis-ready handoff. The candidate table is materialized as
  157160 generated promoter rows from `background_only`, `ethanol`,
  `ciprofloxacin`, and `ethanol_ciprofloxacin` plus 25 measured pDual-10 Reader
  rows, with `latentdna__evo2_7b__context_anchor_mean_bidir_concat` as its
  fixed-length X. The SFXI labels remain separate from the promoted
  response-window Y. Do not synthesize response-window Y from an SFXI score.
- Review round 0 through the pinned label snapshot, RF model artifact, three
  six-row selection sets, allocation trace, and 18-row logical batch. Preserve
  the frozen prediction and allocation evidence before the new constructs are
  measured. Treat the result as a learning probe while `model_support_ready`
  remains false. Physical synthesis still requires a separate study-owned
  handoff and lifecycle decision.
- Use the response metric metastudy `report.md`, generated `review.py`,
  `tables/pressure_tests.csv`, `tables/setpoint_support.csv`,
  `tables/reader_event_intervals.csv`,
  `tables/response_separation_stability.csv`,
  `tables/response_separation_review_scales.csv`,
  `tables/label_model_screen.csv`,
  `tables/retrospective_enrichment_summary.csv`,
  `tables/campaign_greedy_support.csv`,
  `tables/best_fixed_challenger_greedy_support.csv`, and primary plots before
  changing campaign YAMLs or measured-round synthesis handoffs. Do not claim
  ethanol-responsive, ciprofloxacin-responsive, or AND-responsive promoters
  from predicted OPAL scores alone.
- Use the candidate-X scorecard as the pre-assay representation triage surface: bidirectional context-anchor mean is the selected `X`, anchor-source insert mean is the DenseGen-plan baseline, and forward context anchor mean is the strength-standard lens.
- Keep reference-to-plan behavior as a landmark sanity check, not a phenotype claim.
- Keep `native_tf_axis_orientation_audit` as an appendix axis-orientation audit: the generated test supports the LexA/cipro direction and does not support the BaeR/CpxR ethanol direction.
- Use the BioCyc GO sidecars only for source-backed regulator interpretation in appendix enrichment surfaces; keep downstream claims at the level of RegulonDB-associated regulator terms.
- If a linear readout/probe audit is added later, make it a LatentDNA appendix diagnostic with fold-safe preprocessing and no OPAL coupling.
- Re-run `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json` after candidate-table or campaign-state changes and confirm OPAL readiness remains tied to `usr_prom_eth_cip_opal_candidates`, not LatentDNA appendix artifacts.
- Refresh DenseGen plots/notebook if operator-visible DenseGen EDA is needed for study interpretation.
