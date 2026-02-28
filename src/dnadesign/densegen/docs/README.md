## DenseGen documentation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This directory is the DenseGen documentation index. Start here, then open the workflow or reference page you need.

### Documentation by workflow

#### Run a packaged workspace end to end
- [TFBS baseline tutorial](tutorials/demo_tfbs_baseline.md): run the baseline TFBS path from config to outputs.
- [Sampling baseline tutorial](tutorials/demo_sampling_baseline.md): run the sampling-enabled path and inspect artifacts.
- [Constitutive sigma panel tutorial](tutorials/study_constitutive_sigma_panel.md): execute a constitutive sigma study workflow.
- [Stress ethanol and ciprofloxacin tutorial](tutorials/study_stress_ethanol_cipro.md): execute a stress-condition study and verify outputs.

#### Run with Notify
- [DenseGen to USR to Notify tutorial](tutorials/demo_usr_notify.md): run event-producing flow across DenseGen, USR, and Notify.
- [Observability and events](concepts/observability_and_events.md): interpret run events, status transitions, and emitted metadata.

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
- [Packaged workspace catalog](../workspaces/catalog.md): available packaged workspaces and their usage notes.

### Documentation by type
- [docs index](index.md): type-based index for concept, how-to, tutorial, and reference docs.
- [tutorials/](tutorials/): executable end-to-end walkthroughs.
- [howto/](howto/): operational runbooks for environment-specific execution.
- [concepts/](concepts/): behavioral models and lifecycle explanations.
- [reference/](reference/): stable schema, CLI, and artifact contracts.
- [dev/](dev/): maintainer architecture notes and journal entries.
