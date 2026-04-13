# stress_ethanol_cipro_growth latentdna workspace

This workspace is the study-bound `latentdna` scaffold for `docs/studies/stress_ethanol_cipro_growth`.

- Sources are already bound to the checked-in promoter handoff datasets.
- `study_binding` points back to the study record so validation and status surfaces can reconcile readiness.
- The workspace is intentionally scaffold-only right now: the config validates against the live datasets, but no plot, notebook, cluster, or export artifacts have been materialized under `outputs/latentdna` yet.

Primary next steps:

1. `uv run latentdna validate workspace --workspace src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth --deep`
2. `uv run latentdna deliverable run atlas_2x2_intermediate_main --workspace src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth`
3. `uv run latentdna notebook smoke --workspace src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth`
