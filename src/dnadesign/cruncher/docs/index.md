## Cruncher docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-26


### Contents
- [Overview](#overview)
- [Choose your path](#choose-your-path)
- [Docs map](#docs-map)
- [Demos](#demos)
- [Guides](#guides)
- [References](#references)
- [Internals and dev notes](#internals-and-dev-notes)

### Overview
This index is the browsing map for Cruncher docs. Start with the package-level [Cruncher README](../README.md) for the short operational map, then route into one workflow family.

Cruncher has three first-class workflow families today:

- fixed-length optimization workspaces built around `fetch -> lock -> parse -> sample -> analyze`
- cassette workspaces built around `cassette init-workspace|validate|design|solve|show`
- YIU protocol-state workspaces built around `yiu init-workspace|validate|design|trace|show`

Studies and portfolios extend the fixed-length optimization lane. Cassette and YIU runs keep separate workspace and artifact contracts. More families can be added beside these lanes without rewriting the existing ones.

### Choose your path
- Run the core fixed-length optimization lane: [`demos/demo_pairwise.md`](demos/demo_pairwise.md)
- Run the three-TF optimization lane: [`demos/demo_multitf.md`](demos/demo_multitf.md)
- Run the larger study-ready optimization workspace: [`demos/project_all_tfs.md`](demos/project_all_tfs.md)
- Run a cassette workspace end to end: [`demos/demo_cassette_workspace.md`](demos/demo_cassette_workspace.md)
- Run a YIU workspace end to end: [`demos/demo_yiu_workspace.md`](demos/demo_yiu_workspace.md)
- Design a cassette from an authored spec: [`guides/cassette_workflow.md`](guides/cassette_workflow.md)
- Search for ranked cassette hits: [`guides/cassette_solve_workflow.md`](guides/cassette_solve_workflow.md)
- Design a YIU protocol-state workflow: [`guides/yiu_workflow.md`](guides/yiu_workflow.md)
- Understand sample/Gibbs analysis outputs: [`guides/sampling_and_analysis.md`](guides/sampling_and_analysis.md)
- Run study sweeps: [`guides/studies.md`](guides/studies.md)
- Aggregate a portfolio: [`guides/portfolio_aggregation.md`](guides/portfolio_aggregation.md)
- Debug failures quickly: [`guides/troubleshooting.md`](guides/troubleshooting.md)

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

### Demos
- [`demos/demo_pairwise.md`](demos/demo_pairwise.md)
- [`demos/demo_multitf.md`](demos/demo_multitf.md)
- [`demos/project_all_tfs.md`](demos/project_all_tfs.md)
- [`demos/demo_cassette_workspace.md`](demos/demo_cassette_workspace.md)
- [`demos/demo_yiu_workspace.md`](demos/demo_yiu_workspace.md)

### Guides
- [`guides/cassette_workflow.md`](guides/cassette_workflow.md)
- [`guides/cassette_solve_workflow.md`](guides/cassette_solve_workflow.md)
- [`guides/yiu_workflow.md`](guides/yiu_workflow.md)
- [`guides/intent_and_lifecycle.md`](guides/intent_and_lifecycle.md)
- [`guides/ingestion.md`](guides/ingestion.md)
- [`guides/meme_suite.md`](guides/meme_suite.md)
- [`guides/sampling_and_analysis.md`](guides/sampling_and_analysis.md)
- [`guides/studies.md`](guides/studies.md)
- [`guides/study_length_vs_score.md`](guides/study_length_vs_score.md)
- [`guides/study_diversity_vs_score.md`](guides/study_diversity_vs_score.md)
- [`guides/portfolio_aggregation.md`](guides/portfolio_aggregation.md)
- [`guides/troubleshooting.md`](guides/troubleshooting.md)

### References
- [`reference/config.md`](reference/config.md)
- [`reference/cli.md`](reference/cli.md)
- [`reference/architecture.md`](reference/architecture.md)
- [`reference/artifacts.md`](reference/artifacts.md)
- [`reference/cassette_spec.md`](reference/cassette_spec.md)
- [`reference/cassette_solve_spec.md`](reference/cassette_solve_spec.md)
- [`reference/nickase_catalog.md`](reference/nickase_catalog.md)
- [`reference/cassette_artifacts.md`](reference/cassette_artifacts.md)
- [`reference/yiu_spec.md`](reference/yiu_spec.md)
- [`reference/yiu_artifacts.md`](reference/yiu_artifacts.md)
- [`reference/glossary.md`](reference/glossary.md)
- [`reference/doc_conventions.md`](reference/doc_conventions.md)
- [`reference/runbook_steps.md`](reference/runbook_steps.md)

### Internals and dev notes
- [`internals/spec.md`](internals/spec.md)
- [`internals/optimizer_improvements_plan.md`](internals/optimizer_improvements_plan.md)
- [`dev/journal.md`](dev/journal.md)
- [`meta/style_guide.md`](meta/style_guide.md)
