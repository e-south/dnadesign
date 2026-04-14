## stress_ethanol_cipro_growth

- Last verified: 2026-04-13
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

- LatentDNA: `configured`; the study-bound workspace exists and current readiness is `missing` because no downstream artifacts are materialized yet
- Cluster: `planned`; no study-owned results root is configured yet
- OPAL: `not configured`; no study-owned campaign config is checked in yet
- Use `routes.md` for owner tool, entry artifact, primary doc or workspace, and first command per downstream branch

### Next actions

- Use `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json` before submitting or resuming live Infer work
- Keep Infer as lane-specific work: start from `anchor_only` or `anchor_plus_template` lane configs or the matching notify-enabled batch presets
- Materialize the LatentDNA scaffold only after the Infer outputs you want to analyze are stable
- Do not treat Cluster or OPAL as active until a canonical feature matrix or explicit export surface is checked in
