## DenseGen documentation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

### Documentation by workflow

#### Run a packaged workspace end to end
- [TFBS baseline tutorial](tutorials/demo_tfbs_baseline.md): run the baseline TFBS path from config to outputs.
- [Dense array showcase tutorial](tutorials/demo_dense_array_showcase.md): run a local CBC demo with dense TFBS packing and fixed-anchor variants.
- [Sampling baseline tutorial](tutorials/demo_sampling_baseline.md): run the sampling-enabled path and inspect artifacts.
- [Constitutive sigma panel tutorial](tutorials/study_constitutive_sigma_panel.md): execute a constitutive sigma study workflow.
- [Stress ethanol and ciprofloxacin tutorial](tutorials/study_stress_ethanol_cipro.md): execute a stress-condition study and verify outputs.

#### Run with Notify
- [DenseGen to USR to Notify tutorial](tutorials/demo_usr_notify.md): run event-producing flow across DenseGen, USR, and Notify.
- [Observability and events](concepts/observability_and_events.md): interpret run events, status transitions, and emitted metadata.

#### Continue into shared downstream data-plane flows
- [Multi-source shared dataset assembly](../../usr/docs/operations/multi-source-shared-dataset-assembly.md): treat DenseGen outputs as one upstream USR source when construct and infer should share one downstream dataset.
- [Promoter characterization feature matrix](../../usr/docs/operations/promoter-characterization-feature-matrix.md): treat DenseGen outputs as one upstream source when downstream clustering or active learning should consume infer-derived feature columns.

#### Debug a run
1. [Quick checklist](concepts/quick-checklist.md): run preflight checks before executing jobs.
2. [Pipeline lifecycle](concepts/pipeline-lifecycle.md): locate a failing stage and expected transitions.
3. [Outputs and metadata](concepts/outputs-and-metadata.md): verify expected artifacts and metadata surfaces.
4. [CLI reference](reference/cli.md): confirm command contracts and failure semantics.

#### Tune sampling and generation
- [Sampling](concepts/sampling.md): tune sampling behavior and candidate-pool construction.
- [Inputs](concepts/inputs.md): validate source inputs and required normalization rules.
- [Generation](concepts/generation.md): understand generation-stage behavior and constraints.
- [Config reference](reference/config.md): map config keys to runtime behavior.

#### HPC and BU SCC
- [DenseGen HPC runbook](howto/hpc.md): run DenseGen on remote compute with explicit preflight and verify steps.
- [DenseGen on BU SCC](howto/bu-scc.md): BU SCC-specific submission and execution sequence.
- [Repository BU SCC quickstart](../../../../docs/bu-scc/quickstart.md): cluster-level setup and shared operational baseline.

### Workspace documentation
- [DenseGen workspaces directory](../workspaces/README.md): workspace layout and package-local expectations.

### Documentation by type
- [docs index](index.md): type-based index for concept, how-to, tutorial, and reference docs.
- [tutorials/](tutorials/): executable end-to-end walkthroughs.
- [howto/](howto/): operational runbooks for environment-specific execution.
- [concepts/](concepts/): behavioral models and lifecycle explanations.
- [reference/](reference/): stable schema, CLI, and artifact contracts.
- [dev/](dev/): maintainer architecture notes and journal entries.
