## DenseGen

DenseGen wraps the [dense-arrays](https://github.com/e-south/dense-arrays) solver to generate constraint-driven TFBS libraries and write audit-grade artifacts for debugging and downstream analysis. Read this page when you need the fastest route to the right DenseGen document; the detailed operator guidance lives in the docs index.

### Documentation
This section is the canonical starting point for DenseGen docs navigation.

- Start with the **[DenseGen documentation index](docs/index.md)** for the full map of tutorials, how-to guides, concepts, reference, and dev notes.

### Tutorials
This section is for progressive walkthroughs that run end-to-end in workspace order.

- Follow the **[TFBS baseline tutorial](docs/tutorials/demo_tfbs_baseline.md)** for the smallest lifecycle run.
- Follow the **[sampling baseline tutorial](docs/tutorials/demo_sampling_baseline.md)** for Stage-A mining and Stage-B sampling.
- Follow the **[DenseGen to USR to Notify tutorial](docs/tutorials/demo_usr_notify.md)** for event-driven operations.
- Follow the **[constitutive sigma panel study tutorial](docs/tutorials/study_constitutive_sigma_panel.md)** for combinatorial promoter panel workflows.

### How-to guides
This section is for task-oriented runbooks when you already know the basics.

- Use the **[Cruncher to DenseGen PWM handoff guide](docs/howto/cruncher_pwm_pipeline.md)** when preparing motif artifacts for DenseGen.
- Use the **[HPC runbook](docs/howto/hpc.md)** for scheduler-safe run and resume patterns.
- Use the **[BU SCC delta guide](docs/howto/bu-scc.md)** for BU-specific submission details.

### Concepts
This section is for mental models and lifecycle semantics that explain why DenseGen behaves the way it does.

- Read **[observability and events](docs/concepts/observability_and_events.md)** for the DenseGen diagnostics and USR event boundary.
- Read **[pipeline stages and lifecycle](docs/concepts/operator-arc.md)** for Stage-A, Stage-B, solve, and post-run boundaries.

### Reference
This section is for contract-grade details and exact command/schema surfaces.

- Use the **[CLI reference](docs/reference/cli.md)** for command and flag contracts.
- Use the **[config reference](docs/reference/config.md)** for schema keys, strict validation, and examples.
- Use the **[outputs reference](docs/reference/outputs.md)** for artifact and event contracts.

### Developer notes
This section is for maintainers working on DenseGen internals.

- Read the **[DenseGen architecture notes](docs/dev/architecture.md)** for module boundaries.
- Read the **[DenseGen dev journal](docs/dev/journal.md)** for implementation history and rationale.
