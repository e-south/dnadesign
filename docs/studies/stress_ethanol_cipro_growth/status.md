## stress_ethanol_cipro_growth

- Last verified: 2026-05-09
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Route map: `routes.md`
- Study execution map: `pipeline.yaml`
- LatentDNA binding: `latentdna_binding.yaml`
- USR root: `src/dnadesign/usr/datasets`
- Snapshot posture: current after local pull from BU SCC `cluster`
- Preflight posture: available; supported Evo2 7B Infer sequence-view lanes are complete, so the next execution-readiness question is downstream LatentDNA review or optional DenseGen plot refresh, not GPU Infer submission.

### Current Datasets

- DenseGen anchor source: `densegen_prom_eth_cip_source` (`present`, 157160 rows)
- Native/reference promoter source: `usr_promoter_references` (`present`, 48 rows)
- SFXI pDual-10 DenseGen source: `usr_sfxi_pdual10_densegen_promoters` (`present`, 23 rows)
- Construct template seed: `usr_pdual10_plasmid_template` (`present`, 1 row)
- Anchor-only handoff: `usr_prom_eth_cip_anchor` (`present`, 157279 rows)
- Full-context handoff: `construct_prom_eth_cip_context` (`present`, 314558 rows)
- Reference core60 handoff: `construct_prom_eth_cip_reference_core60` (`present`, 48 rows)
- Reference context handoff: `construct_prom_eth_cip_reference_contexts` (`present`, 96 rows)
- RegulonDB native promoter source: `usr_regulondb_native_promoters` (`present`, 3182 rows; regulatory-interaction sidecar must be refreshed before the TF-axis audit)
- RegulonDB native core60 source: `usr_regulondb_native_promoter_core60` (`present`, 3181 rows; planned append into the existing `usr_prom_eth_cip_anchor` study quota)
- Canonical consolidated feature dataset: `usr_prom_eth_cip_matrix` (`planned`)
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
- Current next surface: `src/dnadesign/latentdna/docs/workflows/promoter-study-representation-comparison.md`
- Current working pre-assay `X`: `intermediate_embedding_7b_context_anchor_mean_bidir_concat`
- Conservative DenseGen-plan baseline: `intermediate_embedding_7b_anchor_60bp`
- Strength-standard interpretation lens: `intermediate_embedding_7b_full_context_anchor_mean`

The study is still pre-assay representation triage. LatentDNA has promoted a
working pre-assay candidate `X` for downstream planning, but it has not promoted
a phenotype-validated final `X`.

### Current Infer Coverage

- Sequence-view product contracts: `4/4 ok`
- Infer feature-completion checks: `2/2 ok`
- Required 7B feature views: `786635`
- Reusable 7B vectors: `1573270`
- Reusable 7B scalars: `943962`
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

`fill-infer --no-submit` now skips the six supported complete lanes, skips the three unsupported or retired lanes, and plans no runnable GPU jobs for the current row quota. After the planned RegulonDB core60 append to `usr_prom_eth_cip_anchor` and `construct_prom_eth_cip_context`, the same 7B sequence-view lanes should plan only the newly missing native audit rows.

### Current Downstream Posture

- DenseGen analysis surface: `attention`; the source dataset is ready, but the operator-visible plot inventory contains stale artifacts and should be refreshed before relying on DenseGen plots as current.
- LatentDNA: `attention`; the native TF-axis route is now configured as a planned first-class overlay over the existing study context view, but generated view rows/plots/notebook outputs have not been regenerated after the lineage-metadata config change.
- LatentDNA native TF-axis overlay: `planned`; the workspace now declares the first-class audit deliverable over the existing context-anchor bidirectional view, but it should render only after RegulonDB native core60 rows are appended through `usr_prom_eth_cip_anchor` and `construct_prom_eth_cip_context`, regulatory interactions are populated, and matching 7B feature sidecars exist.
- Cluster: `planned`; use `routes.md` for the current exploratory-clustering handoff.
- OPAL: `not_configured`; no active OPAL campaign has been chosen.

Current LatentDNA decision surfaces:

- `dataset_overview`
- `design_structure_summary`
- `sigma35_ordinal_audit`
- `context_robustness_summary`
- `candidate_decision_frontier`
- `candidate_x_selection_scorecard`

LatentDNA gate:

- `representation_health_summary`

LatentDNA primary review path:

- `dataset_overview`
- `design_structure_summary`
- `sigma35_ordinal_audit`
- `context_robustness_summary`
- `candidate_decision_frontier`
- `candidate_x_selection_scorecard`

LatentDNA companion visuals:

- `balanced_design_family_margin_gallery`
- `sigma35_margin_ladder_gallery`
- `sigma35_stress_margin_gallery`
- `context_pair_summary`
- `reference_to_plan_centroid_heatmap`
- `reference_standard_strength_audit`

LatentDNA appendix support:

- `sigma35_centroid_distance_gallery`
- `native_tf_axis_orientation_audit` (`planned` until the RegulonDB core60 quota append and matching 7B sidecars are complete)
- `appendix_geometry_review`
- `appendix_umap_gallery`

The current checked-in browser artifacts are limited to the previously
available 7B sequence-view feature sidecars. The default browser geometries
include the controlled equal-block bidirectional forward/RC anchor-mean
candidate. Appendix deliverables remain secondary review material, not the
evidence source for selecting `X`. Regenerate LatentDNA outputs after the
metadata/config update before treating the checked-in notebook as current.

Pooling semantics guardrail: Infer mean-pools over token positions. Because
Evo2 token states are causal in the emitted orientation, `anchor_mean` is a
prefix-conditioned anchor-span mean from a full-sequence pass. The
forward/reverse-complement concat is therefore best described as an
equal-block, two-orientation 1 kb context-anchor summary. It is analogous to
the standard forward-plus-reverse workaround for causal sequence models, but it
is not a native bidirectional Evo2 state or hidden state.

### Next Actions

- Use the candidate-X scorecard as the current pre-assay representation triage surface: bidirectional context-anchor mean is the working `X`, anchor-source insert mean is the DenseGen-plan baseline, and forward context anchor mean is the strength-standard lens.
- Keep reference-to-plan behavior as a landmark sanity check, not a phenotype claim.
- Refresh `usr_regulondb_native_promoters/_relations/regulatory_interactions.parquet`, append `usr_regulondb_native_promoter_core60` through the existing `usr_prom_eth_cip_anchor` and `construct_prom_eth_cip_context` handoff, and then run `fill-infer --no-submit` to plan only the new 7B sidecar work before rendering `native_tf_axis_orientation_audit`.
- Re-materialize the affected LatentDNA views and re-run `uv run latentdna workspace snapshot --workspace stress_ethanol_cipro_growth --json` after the native lineage metadata/config update.
- Re-run `uv run ops progress show usr.data-plane.promoter-study-status --json` after regeneration and confirm the LatentDNA attention flag clears or is still explained by a concrete generated-artifact gap.
- Refresh DenseGen plots/notebook if operator-visible DenseGen EDA is needed for current study interpretation.
