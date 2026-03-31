## Cruncher Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-27


**Last updated by:** cruncher-maintainers on 2026-03-27

Cruncher currently has three first-class workflow families:

- fixed-length PWM optimization workspaces built around `fetch -> lock -> parse -> sample -> analyze`
- cassette workspaces built around `cassette init-workspace|validate|design|solve|show`
- YIU protocol-state workspaces built around `yiu init-workspace|validate|trace|solve|show|render`

Studies and portfolios build on the fixed-length optimization lane. Cassette and YIU runs keep their own workspace and artifact contracts. Future lanes should sit beside these families rather than overload them.

### Progressive disclosure route
1. Choose one workflow family under **Choose a workflow family**.
2. Start with one demo or guide from that family.
3. Use **Reference contracts** when you need strict CLI, schema, or artifact behavior.

<!-- docs:toc:off -->

### Choose a workflow family

- **Fixed-length optimization workspaces:** start with [Pairwise Demo](demos/demo_pairwise.md), then move to [Sampling and Analysis](guides/sampling_and_analysis.md) and [Intent and Lifecycle](guides/intent_and_lifecycle.md).
- **Cassette workspaces:** start with [Cassette Workspace Demo](demos/demo_cassette_workspace.md), then use [Cassette Workflow](guides/cassette_workflow.md) or [Cassette Solve Workflow](guides/cassette_solve_workflow.md).
- **YIU protocol-state workspaces:** start with [YIU Workspace Demo](demos/demo_yiu_workspace.md), then use [YIU Workflow](guides/yiu_workflow.md).
- **Study and aggregation workflows:** use [Studies](guides/studies.md) for workspace-scoped sweeps and [Portfolio Aggregation](guides/portfolio_aggregation.md) for cross-workspace handoff packages.
- **Reference contracts:** use [CLI Reference](reference/cli.md), [Architecture](reference/architecture.md), and the relevant schema/artifact reference for your lane.

### Documentation by workflow
<!-- docs:map:start -->
#### Optimize Fixed-Length Sequences
- [Pairwise Demo](demos/demo_pairwise.md)
- [MultiTF Demo](demos/demo_multitf.md)
- [Project Workspace Demo](demos/project_all_tfs.md)
- [Intent and Lifecycle](guides/intent_and_lifecycle.md)
- [Sampling and Analysis](guides/sampling_and_analysis.md)
- [Ingestion](guides/ingestion.md)
- [MEME Suite](guides/meme_suite.md)
- [Artifacts Reference](reference/artifacts.md)

#### Design and Search Cassettes
- [Cassette Workspace Demo](demos/demo_cassette_workspace.md)
- [Cassette Workflow](guides/cassette_workflow.md)
- [Cassette Solve Workflow](guides/cassette_solve_workflow.md)
- [Cassette Spec Reference](reference/cassette_spec.md)
- [Cassette Solve Spec Reference](reference/cassette_solve_spec.md)
- [Nickase Catalog Reference](reference/nickase_catalog.md)
- [Cassette Artifacts](reference/cassette_artifacts.md)

#### Model YIU Hairpin Oligo Processing
- [YIU Workspace Demo](demos/demo_yiu_workspace.md)
- [YIU Workflow](guides/yiu_workflow.md)
- [YIU Spec Reference](reference/yiu_spec.md)
- [YIU Artifacts](reference/yiu_artifacts.md)

#### Run Studies and Portfolio Aggregation
- [Studies](guides/studies.md)
- [Study Length vs Score](guides/study_length_vs_score.md)
- [Study Diversity vs Score](guides/study_diversity_vs_score.md)
- [Portfolio Aggregation](guides/portfolio_aggregation.md)

#### Troubleshooting and Support
- [Troubleshooting](guides/troubleshooting.md)

#### Reference contracts
- [Config Reference](reference/config.md)
- [CLI Reference](reference/cli.md)
- [Architecture](reference/architecture.md)
- [Cassette Spec Reference](reference/cassette_spec.md)
- [Cassette Solve Spec Reference](reference/cassette_solve_spec.md)
- [Nickase Catalog Reference](reference/nickase_catalog.md)
- [Cassette Artifacts](reference/cassette_artifacts.md)
- [YIU Spec Reference](reference/yiu_spec.md)
- [YIU Artifacts](reference/yiu_artifacts.md)
- [Glossary](reference/glossary.md)
- [Runbook Step Reference](reference/runbook_steps.md)
- [Doc Conventions](reference/doc_conventions.md)

#### Maintainer Internals
- [Cruncher Internals Spec](internals/spec.md)
- [Optimizer Improvements Plan](internals/optimizer_improvements_plan.md)
- [Dev Journal](dev/journal.md)
- [Docs Style Guide](meta/style_guide.md)
<!-- docs:map:end -->
