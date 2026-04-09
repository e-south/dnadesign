## Cruncher docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-07

**Last updated by:** cruncher-maintainers on 2026-04-07

### Contents
- [Documentation map](#documentation-map)

### Documentation map
<!-- docs:map:start -->
#### Start with a packaged demo
- [Pairwise Demo](demos/demo_pairwise.md), [MultiTF Demo](demos/demo_multitf.md), and [Project Workspace Demo](demos/project_all_tfs.md). Use these when you want a runnable optimization workspace before changing configs or adding studies.

#### Run the fixed-length optimization lane
- [Sampling and Analysis](guides/sampling_and_analysis.md), [Intent and Lifecycle](guides/intent_and_lifecycle.md), [Ingestion](guides/ingestion.md), [MEME Suite](guides/meme_suite.md), and [Artifacts Reference](reference/artifacts.md). These pages cover the core fetch, lock, parse, sample, analyze flow and the artifacts it emits.

#### Design or search cassettes
- [Cassette Workspace Demo](demos/demo_cassette_workspace.md), [Cassette Workflow](guides/cassette_workflow.md), [Cassette Solve Workflow](guides/cassette_solve_workflow.md), [Cassette Spec Reference](reference/cassette_spec.md), [Cassette Solve Spec Reference](reference/cassette_solve_spec.md), [Nickase Catalog Reference](reference/nickase_catalog.md), and [Cassette Artifacts](reference/cassette_artifacts.md). Start with the demo for a runnable workspace, then move to the guide or reference that matches your task.

#### Run YIU workflows
- [YIU Workspace Demo](demos/demo_yiu_workspace.md), [YIU Workflow](guides/yiu_workflow.md), [YIU Spec Reference](reference/yiu_spec.md), [YIU Artifacts](reference/yiu_artifacts.md), and [YIU Visual System](reference/yiu_visual_system.md). Use the demo to start from a checked-in workspace, the workflow guide for the operator path, and the references when you need schema or bundle details.

#### Reuse Sample outputs in YIU
- [demo_monotypic_tetr runbook](../workspaces/demo_monotypic_tetr/runbook.md) and [demo_monotypic_lexa runbook](../workspaces/demo_monotypic_lexa/runbook.md). These examples show YIU handoffs that stay beside the upstream Sample workspaces that produced the source payloads.

#### Run studies and portfolio aggregation
- [Studies](guides/studies.md), [Study Length vs Score](guides/study_length_vs_score.md), [Study Diversity vs Score](guides/study_diversity_vs_score.md), and [Portfolio Aggregation](guides/portfolio_aggregation.md). These pages cover parameter sweeps and cross-workspace handoff packages built on optimization outputs.

#### Look up tool-wide contracts
- [CLI Reference](reference/cli.md), [Architecture](reference/architecture.md), [Config Reference](reference/config.md), [Glossary](reference/glossary.md), [Runbook Step Reference](reference/runbook_steps.md), and [Doc Conventions](reference/doc_conventions.md). Use these when you need exact command behavior, config fields, artifact paths, or terminology.

#### Troubleshoot a failing run
- [Troubleshooting](guides/troubleshooting.md). Start here when you need the shortest path from a symptom to the relevant lane-specific fix.

#### Maintainer internals
- [Cruncher Internals Spec](internals/spec.md), [Optimizer Improvements Plan](internals/optimizer_improvements_plan.md), [Dev Journal](dev/journal.md), and [Docs Style Guide](meta/style_guide.md). These pages are maintainer-facing and not part of the main operator path.
<!-- docs:map:end -->
