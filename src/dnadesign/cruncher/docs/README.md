## Cruncher Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


**Last updated by:** cruncher-maintainers on 2026-02-23

### Progressive disclosure route
1. Start with one demo from **Run End-to-End Workflows**.
2. Move to one guide in **Optimize and Analyze Outputs** for your immediate task.
3. Use **Reference Contracts** when you need strict schema, CLI, or artifact details.

<!-- docs:toc:off -->

### Documentation by workflow
<!-- docs:map:start -->
#### Run End-to-End Workflows
- [Pairwise Demo](demos/demo_pairwise.md): run the two-TF workflow from fetch to analysis artifacts.
- [MultiTF Demo](demos/demo_multitf.md): run the multi-TF workflow with full output surfaces.
- [Project Workspace Demo](demos/project_all_tfs.md): execute workspace-scale runs across all configured TFs.

#### Design Dual-Context Cassettes
- [Cassette Workflow](guides/cassette_workflow.md): validate and materialize a dual-context hairpin cassette spec.
- [Cassette Solve Workflow](guides/cassette_solve_workflow.md): search for ranked cassette hits and materialize top candidates.
- [Cassette Spec Reference](reference/cassette_spec.md): authoritative cassette schema and invariant semantics.
- [Cassette Solve Spec Reference](reference/cassette_solve_spec.md): solve-schema and search guardrail contract.
- [Nickase Catalog Reference](reference/nickase_catalog.md): local nickase catalog schema and cut-offset rules.
- [Cassette Artifacts](reference/cassette_artifacts.md): deterministic output layout and report files.

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
- [Cassette Spec Reference](reference/cassette_spec.md): cassette schema and coordinate semantics.
- [Cassette Solve Spec Reference](reference/cassette_solve_spec.md): solve schema, preset catalog use, and search limits.
- [Nickase Catalog Reference](reference/nickase_catalog.md): nickase entry schema and validation rules.
- [Cassette Artifacts](reference/cassette_artifacts.md): cassette output files and manifest semantics.
- [Glossary](reference/glossary.md): shared vocabulary for models, metrics, and artifacts.
- [Runbook Step Reference](reference/runbook_steps.md): shared runbook stage names and meanings.
- [Doc Conventions](reference/doc_conventions.md): documentation structure and writing contracts.

#### Maintainer Internals
- [Cruncher Internals Spec](internals/spec.md): implementation-level behavior and invariants.
- [Optimizer Improvements Plan](internals/optimizer_improvements_plan.md): active optimization design backlog and rationale.
- [Dev Journal](dev/journal.md): maintainer investigations, decisions, and validation notes.
- [Docs Style Guide](meta/style_guide.md): style rules for sustaining docs consistency.
<!-- docs:map:end -->
