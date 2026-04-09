## Cruncher docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-07

**Last updated by:** cruncher-maintainers on 2026-04-07


### Contents
- [Overview](#overview)
- [Workflow families](#workflow-families)
- [Docs map](#docs-map)

### Overview
Cruncher has three workflow families:

- fixed-length optimization workspaces built around `fetch -> lock -> parse -> sample -> analyze`
- cassette workspaces built around `cassette init-workspace|validate|design|solve|show`
- payload-centric YIU workspaces for payload validation, rendering, and bundle inspection

Studies and portfolio aggregation extend the fixed-length optimization lane. Use the package-level [Cruncher README](../README.md) for the short package overview.

### Workflow families
- **Fixed-length optimization:** [`demos/demo_pairwise.md`](demos/demo_pairwise.md), [`demos/demo_multitf.md`](demos/demo_multitf.md), [`demos/project_all_tfs.md`](demos/project_all_tfs.md), [`guides/sampling_and_analysis.md`](guides/sampling_and_analysis.md), [`guides/studies.md`](guides/studies.md), and [`guides/portfolio_aggregation.md`](guides/portfolio_aggregation.md)
- **Cassette workflows:** [`demos/demo_cassette_workspace.md`](demos/demo_cassette_workspace.md), [`guides/cassette_workflow.md`](guides/cassette_workflow.md), [`guides/cassette_solve_workflow.md`](guides/cassette_solve_workflow.md), [`reference/cassette_spec.md`](reference/cassette_spec.md), and [`reference/cassette_artifacts.md`](reference/cassette_artifacts.md)
- **Payload-Centric YIU Workflows:** [`demos/demo_yiu_workspace.md`](demos/demo_yiu_workspace.md), [`guides/yiu_workflow.md`](guides/yiu_workflow.md), [`reference/yiu_spec.md`](reference/yiu_spec.md), [`reference/yiu_artifacts.md`](reference/yiu_artifacts.md), and [`reference/yiu_visual_system.md`](reference/yiu_visual_system.md)
- **Sample-backed YIU examples:** [`../workspaces/demo_monotypic_tetr/runbook.md`](../workspaces/demo_monotypic_tetr/runbook.md) and [`../workspaces/demo_monotypic_lexa/runbook.md`](../workspaces/demo_monotypic_lexa/runbook.md)
- **Tool-wide references:** [`reference/cli.md`](reference/cli.md), [`reference/architecture.md`](reference/architecture.md), [`reference/config.md`](reference/config.md), [`reference/glossary.md`](reference/glossary.md), [`reference/runbook_steps.md`](reference/runbook_steps.md), and [`guides/troubleshooting.md`](guides/troubleshooting.md)

### Docs map
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
