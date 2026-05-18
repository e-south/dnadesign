## stress_ethanol_cipro_growth

- Last verified: 2026-05-14
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Route map: `../routes/README.md`
- Study execution map: `../operations/runtime/command-groups/pipeline.yaml`
- LatentDNA binding: `../contexts/latentdna/binding.yaml`
- USR root: `src/dnadesign/usr/datasets`
- Snapshot posture: current after local pull from BU SCC `cluster`
- Preflight posture: available; supported Evo2 7B Infer sequence-view lanes are complete for the expanded RegulonDB-native audit quota, so the next execution-readiness question is downstream LatentDNA review or optional DenseGen plot refresh, not GPU Infer submission.

### Current Datasets

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
  (`present`, 157160 rows; role `opal_candidate_feature_table`; dense
  generated promoter subset from the broader 160460-row LatentDNA view)
- Logical reference feature entry: `infer_prom_eth_cip_reference_views_7b` (`planned`, not separately materialized; current payloads live in dataset-local `_derived/infer/` sidecars)

### Current Phase

- Declared phase: `latentdna_reference_normalization_audit`
- Superseded note: previous study prose said, "The study phase is `infer_batch_preparation`"; current local status has advanced to `latentdna_reference_normalization_audit` after the completed 7B Infer sidecar refresh.
- DenseGen growth: `parallel_optional`
- Merged anchor set: `complete`
- Construct context expansion: `complete`
- Evo2 7B sequence-view Infer sidecars: `complete`
- Preferred infer family: `evo2_7b`
- Supported infer families: `evo2_7b`, `evo2_20b`
- Secondary/debug-required family: `evo2_20b`
- LatentDNA browser default family: `evo2_7b`
- Current next surface: `src/dnadesign/latentdna/docs/workflows/stress-ethanol-cipro-representation-comparison.md`
- Current working pre-assay `X`: `intermediate_embedding_7b_context_anchor_mean_bidir_concat`
- Conservative DenseGen-plan baseline: `intermediate_embedding_7b_anchor_60bp`
- Strength-standard interpretation lens: `intermediate_embedding_7b_full_context_anchor_mean`

The study is still pre-assay representation triage. LatentDNA has promoted a
working pre-assay candidate `X` for downstream planning, but it has not promoted
a phenotype-validated final `X`.

### Current Infer Coverage

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

`fill-infer --no-submit` now skips the six supported complete lanes, skips the three unsupported or retired lanes, and plans no runnable GPU jobs for the expanded current row quota.

### Current Downstream Posture

- DenseGen analysis surface: `attention`; the source dataset is ready, but the operator-visible plot inventory contains stale artifacts and should be refreshed before relying on DenseGen plots as current.
- LatentDNA: `attention`; the native TF-axis route is configured as a first-class appendix overlay over the existing study context view, and local view rows/plots/notebook outputs have been regenerated, but `design_structure_summary` and `sigma35_ordinal_audit` remain pending before the full primary review path is current.
- LatentDNA native TF-axis overlay: `current`; the deliverable renders over the existing context-anchor bidirectional view after RegulonDB native core60 rows were appended through `usr_prom_eth_cip_anchor` and `construct_prom_eth_cip_context`, regulatory interactions were populated, and matching 7B feature sidecars were filled.
- RegulonDB functional annotation sidecars: `current`; `usr_regulondb_native_promoters/_relations/` now carries BioCyc KB 29.6 regulator GO terms, promoter-regulator-GO term rows, and regulator coverage rows. LatentDNA now has a separate BioCyc GO biological-process appendix plot that reuses the native plan-margin tail tables for interpretation. These are source-backed annotation sidecars, not OPAL inputs or mechanistic labels.
- Cluster: `planned`; use `../routes/README.md` for the current exploratory-clustering handoff.
- OPAL: `candidate_table_materialized_pre_assay`; batch-0 campaign configs
  exist for ethanol factor, ciprofloxacin factor, and AND objectives, and the
  shared candidate feature table is materialized. The observed-label sidecar
  and campaign state remain absent until round-0 assay labels are ingested.

Current LatentDNA decision surfaces: see
`../contexts/latentdna/review-surfaces.md` for the review-surface inventory,
browser-control semantics, UMAP caveats, and pooling guardrails.

### Next Actions

- Validate the three OPAL campaign configs against
  `usr_prom_eth_cip_opal_candidates`, then ingest round-0 SFXI labels into the
  shared `_opal/observed_labels.parquet` sidecar once assay data exist. The
  candidate table is already materialized as the dense generated promoter
  subset from `background_only`, `ethanol`, `ciprofloxacin`, and
  `ethanol_ciprofloxacin`, excluding archive SFXI/reference/control rows, with
  `latentdna__evo2_7b__context_anchor_mean_bidir_concat` as its fixed-length X.
- Use the candidate-X scorecard as the current pre-assay representation triage surface: bidirectional context-anchor mean is the working `X`, anchor-source insert mean is the DenseGen-plan baseline, and forward context anchor mean is the strength-standard lens.
- Keep reference-to-plan behavior as a landmark sanity check, not a phenotype claim.
- Keep `native_tf_axis_orientation_audit` as an appendix axis-orientation audit: the current generated test supports the LexA/cipro direction and does not support the BaeR/CpxR ethanol direction.
- Use the BioCyc GO sidecars only for source-backed regulator interpretation in appendix enrichment surfaces; keep downstream claims at the level of RegulonDB-associated regulator terms.
- If a linear readout/probe audit is added later, make it a LatentDNA appendix diagnostic with fold-safe preprocessing and no OPAL coupling.
- Re-run `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json` after regeneration and confirm the LatentDNA attention flag clears or is still explained by a concrete generated-artifact gap.
- Refresh DenseGen plots/notebook if operator-visible DenseGen EDA is needed for current study interpretation.
