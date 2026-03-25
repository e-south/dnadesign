## Cruncher Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-25


**Last updated by:** cruncher-maintainers on 2026-03-25

Cruncher currently has two first-class workflow families:

- fixed-length PWM optimization workspaces built around `fetch -> lock -> parse -> sample -> analyze`
- cassette workspaces built around `cassette init-workspace|validate|design|solve|show`

Studies and portfolios build on the fixed-length optimization lane. Cassette runs keep their own workspace and artifact contracts. Future lanes should sit beside these families rather than overload them.

### Progressive disclosure route
1. Choose one workflow family under **Choose a workflow family**.
2. Start with one demo or guide from that family.
3. Use **Reference contracts** when you need strict CLI, schema, or artifact behavior.

<!-- docs:toc:off -->

### Choose a workflow family

- **Fixed-length optimization workspaces:** start with [Pairwise Demo](demos/demo_pairwise.md), then move to [Sampling and Analysis](guides/sampling_and_analysis.md) and [Intent and Lifecycle](guides/intent_and_lifecycle.md).
- **Cassette workspaces:** start with [Cassette Workspace Demo](demos/demo_cassette_workspace.md), then use [Cassette Workflow](guides/cassette_workflow.md) or [Cassette Solve Workflow](guides/cassette_solve_workflow.md).
- **Study and aggregation workflows:** use [Studies](guides/studies.md) for workspace-scoped sweeps and [Portfolio Aggregation](guides/portfolio_aggregation.md) for cross-workspace handoff packages.
- **Reference contracts:** use [CLI Reference](reference/cli.md), [Architecture](reference/architecture.md), and the relevant schema/artifact reference for your lane.

### Documentation by workflow
<!-- docs:map:start -->
#### Optimize Fixed-Length Sequences
- [Pairwise Demo](demos/demo_pairwise.md): run the two-TF optimization workflow from fetch to analysis artifacts.
- [MultiTF Demo](demos/demo_multitf.md): run the three-TF optimization workflow with full output surfaces.
- [Project Workspace Demo](demos/project_all_tfs.md): execute the larger workspace flow, including standard study sweeps.
- [Intent and Lifecycle](guides/intent_and_lifecycle.md): understand the fetch/lock/parse/sample/analyze lifecycle and artifact boundaries.
- [Sampling and Analysis](guides/sampling_and_analysis.md): tune Gibbs annealing runs and interpret resulting outputs.
- [Ingestion](guides/ingestion.md): prepare and validate motif inputs before optimization.
- [MEME Suite](guides/meme_suite.md): run MEME/FIMO integration flows and expected artifacts.
- [Artifacts Reference](reference/artifacts.md): verify generated sample/analyze files and schema expectations.

#### Design and Search Cassettes
- [Cassette Workspace Demo](demos/demo_cassette_workspace.md): scaffold a cassette-only workspace, run solve, and render outputs in place.
- [Cassette Workflow](guides/cassette_workflow.md): validate and materialize a dual-context hairpin cassette spec.
- [Cassette Solve Workflow](guides/cassette_solve_workflow.md): search for ranked cassette hits and materialize top candidates.
- [Cassette Spec Reference](reference/cassette_spec.md): authoritative cassette schema and invariant semantics.
- [Cassette Solve Spec Reference](reference/cassette_solve_spec.md): solve schema, selection policy, and search guardrails.
- [Nickase Catalog Reference](reference/nickase_catalog.md): local nickase catalog schema and cut-offset rules.
- [Cassette Artifacts](reference/cassette_artifacts.md): deterministic output layout, views, baserender jobs, and render paths.

#### Run Studies and Portfolio Aggregation
- [Studies](guides/studies.md): orchestrate repeatable study execution loops.
- [Study Length vs Score](guides/study_length_vs_score.md): run and interpret the length-score tradeoff study.
- [Study Diversity vs Score](guides/study_diversity_vs_score.md): run and interpret diversity-score tradeoff study.
- [Portfolio Aggregation](guides/portfolio_aggregation.md): aggregate study outputs for project-level comparison.

#### Troubleshooting and Support
- [Troubleshooting](guides/troubleshooting.md): diagnose common input, runtime, and artifact failures across workflow families.

#### Reference contracts
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
