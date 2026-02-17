## DenseGen

DenseGen wraps the [dense-arrays](https://github.com/e-south/dense-arrays) solver to generate constraint-driven TFBS libraries and write audit-grade artifacts for debugging and downstream analysis. Read this page to pick the right DenseGen document; detailed operator guidance lives in the docs index.

### Documentation
This section is the primary starting point for DenseGen docs navigation.

- Start with the **[DenseGen documentation index](docs/index.md)** for the full map of tutorials, how-to guides, concepts, reference, and dev notes.

### End-to-end usage
This section gives the shortest practical progression from basic run validation to event-driven operations.

- Run **[TFBS baseline tutorial](docs/tutorials/demo_tfbs_baseline.md)** first, then **[sampling baseline tutorial](docs/tutorials/demo_sampling_baseline.md)**.
- Run **[constitutive sigma panel study tutorial](docs/tutorials/study_constitutive_sigma_panel.md)** for plan-template expansion behavior.
- Run **[stress ethanol and ciprofloxacin study tutorial](docs/tutorials/study_stress_ethanol_cipro.md)**, then **[DenseGen to USR to Notify tutorial](docs/tutorials/demo_usr_notify.md)** for watcher flows.

### Tutorials
This section is for progressive walkthroughs that run end-to-end in workspace order.

- Follow the **[TFBS baseline tutorial](docs/tutorials/demo_tfbs_baseline.md)** for `demo_tfbs_baseline` (smallest lifecycle run).
- Follow the **[sampling baseline tutorial](docs/tutorials/demo_sampling_baseline.md)** for `demo_sampling_baseline` (Stage-A mining and Stage-B sampling).
- Follow the **[constitutive sigma panel study tutorial](docs/tutorials/study_constitutive_sigma_panel.md)** for `study_constitutive_sigma_panel` (combinatorial promoter panel workflow).
- Follow the **[stress ethanol and ciprofloxacin study tutorial](docs/tutorials/study_stress_ethanol_cipro.md)** for `study_stress_ethanol_cipro` (three-plan campaign baseline).
- Follow the **[DenseGen to USR to Notify tutorial](docs/tutorials/demo_usr_notify.md)** for event-driven operations.

### How-to guides
This section is for task-oriented runbooks when you already know the basics.

- Use the **[Cruncher to DenseGen PWM handoff guide](docs/howto/cruncher_pwm_pipeline.md)** when preparing motif artifacts for DenseGen.
- Use the **[HPC runbook](docs/howto/hpc.md)** for scheduler-safe run and resume patterns.
- Use the **[DenseGen on BU SCC guide](docs/howto/bu-scc.md)** for BU-specific submission details.

### Concepts
This section is for mental models and lifecycle behavior that explain why DenseGen behaves the way it does.

- Read **[observability and events](docs/concepts/observability_and_events.md)** for the DenseGen diagnostics and USR event boundary.
- Read **[pipeline lifecycle](docs/concepts/pipeline-lifecycle.md)** for Stage-A, Stage-B, solve, and post-run boundaries.

### Reference
This section is for exact commands, schema definitions, and output contracts.

- Use the **[CLI reference](docs/reference/cli.md)** for command and flag contracts.
- Use the **[config reference](docs/reference/config.md)** for schema keys, strict validation, and examples.
- Use the **[outputs reference](docs/reference/outputs.md)** for artifact and event contracts.

### Developer notes
This section is for maintainers working on DenseGen internals.

- Read the **[DenseGen architecture guide](docs/dev/architecture.md)** for module boundaries.
- Read the **[DenseGen dev journal](docs/dev/journal.md)** for implementation history and rationale.
