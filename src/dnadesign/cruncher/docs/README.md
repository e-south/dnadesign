## Cruncher docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-07


**Last updated by:** cruncher-maintainers on 2026-04-07

Cruncher has four workflow families:

- fixed-length optimization workspaces for sequence sampling, analysis, studies, and aggregation
- cassette workspaces for cassette design and ranked search
- snapback workspaces for narrow single-nick foldback validation and bounded search
- payload-centric YIU workspaces for payload validation, rendering, and bundle inspection

Studies and portfolio aggregation build on the fixed-length optimization lane.

### Start here

- **Fixed-length optimization:** [Pairwise Demo](demos/demo_pairwise.md), [Sampling and Analysis](guides/sampling_and_analysis.md), and [Intent and Lifecycle](guides/intent_and_lifecycle.md)
- **Cassette workflows:** [Cassette Workspace Demo](demos/demo_cassette_workspace.md), [Cassette Workflow](guides/cassette_workflow.md), and [Cassette Solve Workflow](guides/cassette_solve_workflow.md)
- **Snapback workflows:** [Snapback Workflow](guides/snapback_workflow.md), [Snapback Artifacts](reference/snapback_artifacts.md), and the checked-in [demo_snapback workspace README](../workspaces/demo_snapback/README.md)
- **Payload-Centric YIU Workflows:** [YIU Workspace Demo](demos/demo_yiu_workspace.md), [YIU Workflow](guides/yiu_workflow.md), [YIU Spec Reference](reference/yiu_spec.md), [YIU Artifacts](reference/yiu_artifacts.md), and [YIU Visual System](reference/yiu_visual_system.md)
- **Sample-backed YIU examples:** [demo_monotypic_tetr runbook](../workspaces/demo_monotypic_tetr/runbook.md) and [demo_monotypic_lexa runbook](../workspaces/demo_monotypic_lexa/runbook.md)
- **Studies and aggregation:** [Studies](guides/studies.md) and [Portfolio Aggregation](guides/portfolio_aggregation.md)
- **Tool-wide references:** [CLI Reference](reference/cli.md), [Architecture](reference/architecture.md), [Config Reference](reference/config.md), [Glossary](reference/glossary.md), and [Runbook Step Reference](reference/runbook_steps.md)

<!-- docs:toc:off -->

### Documentation map
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

#### Validate and Search Single-Nick Foldbacks
- [Snapback Workflow](guides/snapback_workflow.md)
- [Snapback Artifacts](reference/snapback_artifacts.md)
- [demo_snapback README](../workspaces/demo_snapback/README.md)
- [demo_snapback runbook](../workspaces/demo_snapback/runbook.md)

#### Payload-Centric YIU Workflows
- [YIU Workspace Demo](demos/demo_yiu_workspace.md)
- [YIU Workflow](guides/yiu_workflow.md)
- [YIU Spec Reference](reference/yiu_spec.md)
- [YIU Artifacts](reference/yiu_artifacts.md)
- [YIU Visual System](reference/yiu_visual_system.md)

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
- [Snapback Artifacts](reference/snapback_artifacts.md)
- [YIU Spec Reference](reference/yiu_spec.md)
- [YIU Artifacts](reference/yiu_artifacts.md)
- [YIU Visual System](reference/yiu_visual_system.md)
- [Glossary](reference/glossary.md)
- [Runbook Step Reference](reference/runbook_steps.md)
- [Doc Conventions](reference/doc_conventions.md)

#### Maintainer Internals
- [Cruncher Internals Spec](internals/spec.md)
- [Optimizer Improvements Plan](internals/optimizer_improvements_plan.md)
- [Dev Journal](dev/journal.md)
- [Docs Style Guide](meta/style_guide.md)
<!-- docs:map:end -->
