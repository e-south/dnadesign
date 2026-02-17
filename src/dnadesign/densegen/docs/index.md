## DenseGen documentation

DenseGen generates constraint-driven TFBS libraries and produces audit-grade artifacts for solver diagnostics, plotting, notebooks, and downstream event workflows. Read this index when you need a complete map of DenseGen docs and a clear path from first run to advanced operations.

### Contents
This table of contents lets you jump by doc type instead of scanning folders.

- [Doc type definitions](#doc-type-definitions)
- [Tutorials](#tutorials)
- [How-to guides](#how-to-guides)
- [Concepts](#concepts)
- [Reference](#reference)
- [Developer notes](#developer-notes)

### Doc type definitions
This section prevents terminology drift by defining each documentation type in one place.

- **Tutorials** are progressive, end-to-end walkthroughs for learning by doing.
- **How-to guides** are task-oriented runbooks for operators who already know the basics.
- **Concepts** explain lifecycle semantics, architecture boundaries, and mental models.
- **Reference** documents are contract-grade definitions for CLI, schema, and artifacts.

### Tutorials
Tutorials are the fastest way to get productive with DenseGen because they show complete command flows and expected outputs.

- Read the **[TFBS baseline tutorial](tutorials/demo_tfbs_baseline.md)** to run the smallest DenseGen workflow and inspect core artifacts.
- Read the **[sampling baseline tutorial](tutorials/demo_sampling_baseline.md)** to run Stage-A mining, Stage-B sampling, and dual-sink outputs.
- Read the **[DenseGen to USR to Notify tutorial](tutorials/demo_usr_notify.md)** to wire USR events into Notify and verify webhook delivery.
- Read the **[constitutive sigma panel study tutorial](tutorials/study_constitutive_sigma_panel.md)** to exercise promoter-matrix templates and sequence constraints.

### How-to guides
How-to guides are for specific operator tasks where you need focused instructions rather than a full tutorial arc.

- Use the **[Cruncher to DenseGen PWM handoff guide](howto/cruncher_pwm_pipeline.md)** to align motif artifacts with DenseGen Stage-A inputs.
- Use the **[HPC runbook](howto/hpc.md)** for batch-safe preflight, run, resume, and notebook generation patterns.
- Use the **[BU SCC delta guide](howto/bu-scc.md)** for BU-specific scheduler details layered on top of the base HPC runbook.
- Use the **[HPC and Notify legacy redirect](howto/usr_notify_hpc.md)** only when following an older link; it points to current runbooks.
- Use the **[BU SCC end-to-end legacy redirect](howto/bu_scc_end_to_end.md)** only when following an older link; it points to current runbooks.

### Concepts
Concepts explain DenseGen behavior so operators can reason about failures, config choices, and output contracts without trial-and-error.

- Read **[observability and events](concepts/observability_and_events.md)** to understand DenseGen diagnostics, USR mutation events, and Notify boundaries.
- Read **[operator arc](concepts/operator-arc.md)** for the Stage-A to Stage-B to solve lifecycle and run-state behavior.
- Read **[workspace layout](concepts/workspace.md)** to understand workspace structure, reset semantics, and path expectations.
- Read **[workspace templates](concepts/workspace-templates.md)** to understand how packaged demos and studies are intended to scale.
- Read **[inputs](concepts/inputs.md)** to understand Stage-A input types and path resolution.
- Read **[sampling](concepts/sampling.md)** to understand Stage-A candidate selection and Stage-B weighting behavior.
- Read **[generation](concepts/generation.md)** to understand plan constraints, fixed elements, and solver interactions.
- Read **[outputs and metadata](concepts/outputs-metadata.md)** to understand sink layout and join keys.
- Read **[postprocess](concepts/postprocess.md)** to understand pad-mode behavior and GC constraints.
- Read **[quick contracts](concepts/quick-contracts.md)** for a short command-and-artifact checklist.

### Reference
Reference documents are the authoritative contracts for command surfaces, schema keys, and file formats.

- Use the **[CLI reference](reference/cli.md)** to look up exact commands, flags, and behavior.
- Use the **[config reference](reference/config.md)** to look up schema fields and strict validation rules.
- Use the **[outputs reference](reference/outputs.md)** to look up artifact paths, event streams, and notebook/plot surfaces.
- Use the **[motif artifacts reference](reference/motif_artifacts.md)** to look up PWM artifact JSON requirements.

### Developer notes
Developer notes are maintainer-facing documents for code architecture and implementation history.

- Read **[architecture notes](dev/architecture.md)** when tracing pipeline boundaries across CLI, config, runtime, and output modules.
- Read **[development journal](dev/journal.md)** when you need historical rationale for prior implementation decisions.
