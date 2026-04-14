## stress_ethanol_cipro_growth Routes

**Last verified:** 2026-04-13

### DenseGen EDA

- Current state: `parallel_optional`; shared DenseGen source dataset is current at `157160` rows.
- Owner tool: `densegen`
- Entry artifact: `densegen/study_stress_ethanol_cipro`
- Primary doc/workspace: `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/README.md`
- First command: `uv run dense plot -c src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/config.yaml`
- Configured/planned/not configured: `configured`

### Construct lineage

- Current state: `complete`; the shared construct-context dataset is current at `157164` rows.
- Owner tool: `construct`
- Entry artifact: `promoter/stress_ethanol_cipro_anchor_set`
- Primary doc/workspace: `src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10/runbook.md`
- First command: `uv run construct workspace doctor --workspace src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10`
- Configured/planned/not configured: `configured`

### Infer lanes

- Current state: `infer_batch_preparation`; shared handoff datasets are ready and the next execution gate is preflight for the lane-specific Infer presets.
- Owner tool: `infer`
- Entry artifact: `promoter/stress_ethanol_cipro_anchor_set` and `promoter/stress_ethanol_cipro_construct_contexts`
- Primary doc/workspace: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/README.md`
- First command: `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
- Configured/planned/not configured: `configured`

### LatentDNA atlas

- Current state: `scaffold_only`; the study-bound workspace exists, but no atlas, notebook, cluster, or export artifacts are materialized yet.
- Owner tool: `latentdna`
- Entry artifact: `promoter/stress_ethanol_cipro_anchor_set` and `promoter/stress_ethanol_cipro_construct_contexts`
- Primary doc/workspace: `src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/README.md`
- First command: `uv run latentdna validate workspace --workspace src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth --deep`
- Configured/planned/not configured: `configured`

### Cluster exploration

- Current state: `planned`; no study-owned cluster results root is configured yet.
- Owner tool: `cluster`
- Entry artifact: `promoter/stress_ethanol_cipro_feature_matrix` or a later explicit latent export such as `x2_primary_20b`
- Primary doc/workspace: `src/dnadesign/cluster/docs/workflows/exploratory-clustering.md`
- First command: `uv run ops catalog show cluster.downstream.exploratory-clustering`
- Configured/planned/not configured: `planned`

### OPAL campaigns

- Current state: `not configured`; no study-owned OPAL campaign config is checked in yet.
- Owner tool: `opal`
- Entry artifact: `promoter/stress_ethanol_cipro_feature_matrix` or a later explicit latent export such as `x2_primary_20b`
- Primary doc/workspace: `src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md`
- First command: `uv run ops catalog show opal.downstream.usr-infer-x-active-learning`
- Configured/planned/not configured: `not configured`
