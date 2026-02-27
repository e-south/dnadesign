## Cruncher Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


**Last updated by:** cruncher-maintainers on 2026-02-23

<!-- docs:toc:off -->

### Documentation by workflow
<!-- docs:map:start -->
#### Run End-to-End Workflows
- [Pairwise Demo](demos/demo_pairwise.md): run the two-TF workflow from fetch to analysis artifacts.
- [MultiTF Demo](demos/demo_multitf.md): run the multi-TF workflow with full output surfaces.
- [Project Workspace Demo](demos/project_all_tfs.md): execute workspace-scale runs across all configured TFs.

#### Ingest and Prepare Inputs
- [Ingestion](guides/ingestion.md): prepare and validate motif inputs before optimization.
- [MEME Suite](guides/meme_suite.md): run MEME/FIMO integration flows and expected outputs.
- [Troubleshooting](guides/troubleshooting.md): diagnose common input and runtime failures.

#### Optimize and Analyze Outputs
- [Intent and Lifecycle](guides/intent_and_lifecycle.md): understand stage transitions and artifact contracts.
- [Sampling and Analysis](guides/sampling_and_analysis.md): tune optimization settings and interpret results.
- [Artifacts Reference](reference/artifacts.md): verify generated files and schema expectations.

#### Run Studies and Portfolio Aggregation
- [Studies](guides/studies.md): orchestrate repeatable study execution loops.
- [Study Length vs Score](guides/study_length_vs_score.md): run and interpret the length-score tradeoff study.
- [Study Diversity vs Score](guides/study_diversity_vs_score.md): run and interpret diversity-score tradeoff study.
- [Portfolio Aggregation](guides/portfolio_aggregation.md): aggregate study outputs for project-level comparison.

#### Reference Contracts
- [Config Reference](reference/config.md): authoritative configuration schema and field semantics.
- [CLI Reference](reference/cli.md): command/flag contracts and invocation patterns.
- [Architecture](reference/architecture.md): dataflow and module boundaries.
- [Glossary](reference/glossary.md): shared vocabulary for models, metrics, and artifacts.
- [Runbook Step Reference](reference/runbook_steps.md): canonical runbook stage names and meanings.
- [Doc Conventions](reference/doc_conventions.md): documentation structure and writing contracts.

#### Maintainer Internals
- [Cruncher Internals Spec](internals/spec.md): implementation-level behavior and invariants.
- [Optimizer Improvements Plan](internals/optimizer_improvements_plan.md): active optimization design backlog and rationale.
- [Dev Journal](dev/journal.md): maintainer investigations, decisions, and validation notes.
- [Docs Style Guide](meta/style_guide.md): style rules for sustaining docs consistency.
<!-- docs:map:end -->
