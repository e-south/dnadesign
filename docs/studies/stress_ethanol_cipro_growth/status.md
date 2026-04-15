## stress_ethanol_cipro_growth

- Last verified: 2026-04-15
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Route map: `routes.md`
- Study execution map: `pipeline.yaml`
- USR root: `src/dnadesign/usr/datasets`

### Current datasets

- DenseGen anchor source: `densegen/study_stress_ethanol_cipro` (`present`, shared source)
- Wildtype or manual controls: `mg1655_promoters` (`present`, `4` rows)
- Construct template seed: `plasmids` (`present`, `1` row)
- Anchor-only handoff: `promoter/stress_ethanol_cipro_anchor_set` (`present`, shared infer plane)
- Construct-expanded handoff: `promoter/stress_ethanol_cipro_construct_contexts` (`present`, shared infer plane)
- Canonical consolidated feature dataset: `promoter/stress_ethanol_cipro_feature_matrix` (`planned`)

### Current phase

- Declared phase: `infer_batch_preparation`
- DenseGen growth: `parallel_optional`
- Merged anchor set: `complete`
- Construct context expansion: `complete`
- Next in-progress surface: `src/dnadesign/usr/docs/operations/promoter-study-preflight.md`
- Preferred infer family: `evo2_20b`
- Supported infer families: `evo2_20b`, `evo2_7b`

### Current row counts

- `densegen/study_stress_ethanol_cipro`: `157160`
- `promoter/stress_ethanol_cipro_anchor_set`: `157164`
- `promoter/stress_ethanol_cipro_construct_contexts`: `157164`
- `promoter/stress_ethanol_cipro_feature_matrix`: `n/a` (`planned`)
- DenseGen source row target: `100000`
- Current DenseGen row gap: `0`
- Shared handoff metadata posture: `densegen__plan` and `densegen__required_regulators` are complete for all DenseGen-derived handoff rows

### Current downstream posture

- LatentDNA: `configured`; the study-bound workspace validates on the canonical `outputs/` root and the atlas, geometry switchboard, context-audit, agreement, cluster-correspondence, and PCA scree surfaces are materialized. Current readiness is `attention` because the required export bundle `x2_primary_20b` is still missing and the next analysis slice should shift from `delta20`-centric interpretation toward reference alignment, structured anchor-to-context movement, and grouped prediction benchmarking on the pooled representations already saved
- Cluster: `planned`; no study-owned results root is configured yet
- OPAL: `not configured`; no study-owned campaign config is checked in yet
- Use `routes.md` for owner tool, entry artifact, primary doc or workspace, and first command per downstream branch
- Use `uv run ops progress show usr.data-plane.promoter-study-status --json`
  plus `evidence.analysis_surfaces` when you need DenseGen plot ids, LatentDNA
  deliverable ids, notebook paths, or Cluster artifact-layout templates from
  one snapshot

### Next actions

- Use `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json` before submitting or resuming live Infer work
- Use `routes.md` for downstream owner handoff, first commands, and tool-local cleanup or notebook workflow steps
- For LatentDNA, treat `z20_60` and `z20_1k_seq` as the primary study question, keep pooled logits as benchmarks, keep per-base likelihood as a scalar side channel, and treat `z20_1k_anchor`, `drag20`, and Leiden correspondence as QC until the reference-alignment and grouped-benchmark plots are added
- Keep Infer as lane-specific work: start from `anchor_only` or `anchor_plus_template` lane configs or the matching notify-enabled batch presets
- Do not treat Cluster or OPAL as active until a canonical feature matrix or explicit export surface is checked in
