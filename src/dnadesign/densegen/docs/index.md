## DenseGen documentation

DenseGen generates constraint-driven TFBS libraries and audit-grade artifacts for debugging and downstream analysis. Use this index to choose the right document for your task, from first-run tutorials to detailed reference docs.

### Contents
This table of contents lets you jump by doc type instead of scanning folders.

- [Doc type definitions](#doc-type-definitions)
- [End-to-end guidance](#end-to-end-guidance)
- [Tutorials](#tutorials)
- [How-to guides](#how-to-guides)
- [Concepts](#concepts)
- [Reference](#reference)
- [Developer notes](#developer-notes)

### Doc type definitions
This section prevents terminology drift by defining each documentation type in one place.

- **Tutorials**: progressive, end-to-end walkthroughs.
- **How-to guides**: task-focused runbooks for known goals.
- **Concepts**: mental models and lifecycle explanations.
- **Reference**: authoritative contracts for commands, schema, and artifacts.

### End-to-end guidance
This section gives a practical run order so new users can move from smallest demo to campaign-scale flows without guessing.

- Start with **[TFBS baseline tutorial](tutorials/demo_tfbs_baseline.md)**, then move to **[sampling baseline](tutorials/demo_sampling_baseline.md)**.
- For template expansion behavior, run **[constitutive sigma panel study](tutorials/study_constitutive_sigma_panel.md)** after the baselines.
- For Notify-ready campaign flow, run **[stress ethanol and ciprofloxacin study](tutorials/study_stress_ethanol_cipro.md)** and then **[DenseGen to USR to Notify](tutorials/demo_usr_notify.md)**.

### Tutorials
Tutorials are the most direct way to start with DenseGen because they show complete command flows and expected outputs.

- **[TFBS baseline tutorial](tutorials/demo_tfbs_baseline.md)** - `demo_tfbs_baseline`: minimal end-to-end run and core artifacts.
- **[Sampling baseline tutorial](tutorials/demo_sampling_baseline.md)** - `demo_sampling_baseline`: Stage-A mining, Stage-B sampling, dual sinks.
- **[Constitutive sigma panel study tutorial](tutorials/study_constitutive_sigma_panel.md)** - `study_constitutive_sigma_panel`: sigma70 fixed-element combinatorics and sequence constraints.
- **[Stress ethanol and ciprofloxacin study tutorial](tutorials/study_stress_ethanol_cipro.md)** - `study_stress_ethanol_cipro`: three-plan stress campaign with Notify-ready outputs.
- **[DenseGen to USR to Notify tutorial](tutorials/demo_usr_notify.md)** - Event pipeline and watcher validation.

### How-to guides
How-to guides are for specific operator tasks where you need focused instructions rather than a full tutorial arc.

- **[Cruncher to DenseGen PWM handoff](howto/cruncher_pwm_pipeline.md)** - Prepare motif artifacts for Stage-A.
- **[HPC runbook](howto/hpc.md)** - Preflight, run, resume, and post-run analysis on clusters.
- **[DenseGen on BU SCC guide](howto/bu-scc.md)** - BU-specific scheduler and environment differences.
- **[BU SCC quickstart](../../../../docs/bu-scc/quickstart.md)** - Repo-level onboarding and submit patterns.

### Concepts
Concepts explain DenseGen behavior so operators can reason about failures, config choices, and output contracts without trial-and-error.

- **[Observability and events](concepts/observability_and_events.md)** - DenseGen diagnostics vs USR events vs Notify.
- **[Pipeline lifecycle](concepts/pipeline-lifecycle.md)** - Stage lifecycle and run-state behavior.
- **[Workspace layout](concepts/workspace.md)** - Directory model, reset, and resume semantics.
- **[Workspace templates](concepts/workspace-templates.md)** - Demo and study template intent.
- **[Inputs](concepts/inputs.md)** - Stage-A input types and path resolution.
- **[Sampling](concepts/sampling.md)** - Stage-A candidate selection and Stage-B weighting.
- **[Generation](concepts/generation.md)** - Plans, fixed elements, and solver interactions.
- **[Outputs and metadata](concepts/outputs-and-metadata.md)** - Sink layout and join keys.
- **[Postprocess](concepts/postprocess.md)** - Pad behavior and GC constraints.
- **[Quick checklist](concepts/quick-checklist.md)** - Minimal command and artifact checklist.

### Reference
Reference documents are the authoritative contracts for command surfaces, schema keys, and file formats.

- **[CLI reference](reference/cli.md)** - Commands, flags, and behavior.
- **[Config reference](reference/config.md)** - Schema fields and strict validation.
- **[Outputs reference](reference/outputs.md)** - Artifact paths and output contracts.
- **[Motif artifacts reference](reference/motif_artifacts.md)** - PWM artifact JSON contract.

### Developer notes
Developer notes are maintainer-facing documents for code architecture and implementation history.

- **[Architecture notes](dev/architecture.md)** - Pipeline boundaries and key code surfaces.
- **[Development journal](dev/journal.md)** - Historical decisions and validation notes.
