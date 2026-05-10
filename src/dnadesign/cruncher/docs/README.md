## Cruncher docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-23


**Last updated by:** cruncher-maintainers on 2026-04-23

Cruncher currently registers six peer workflow families.
Registered family ids: `sample`, `cassette`, `yiu`, `snapback`, `study`, `portfolio`.

- fixed-length optimization workspaces for sequence sampling and artifact-only analysis
- cassette workspaces for cassette design and ranked search
- payload-centric YIU workspaces for payload validation, rendering, and bundle inspection
- snapback workspaces for preserved-site and released-product single-nick foldback workflows
- study orchestration workspaces for aggregate sweep execution and summary surfaces over explicit source-family runs
- portfolio orchestration workspaces for cross-study aggregation and handoff tables over explicit source-family runs

The `study` and `portfolio` families orchestrate explicit source-family runs.
The currently shipped examples are sample-backed, but the orchestration family
boundaries remain separate command surfaces with their own workspace contracts
and docs surfaces.

### Start here

- **Fixed-length optimization:** [Pairwise Demo](demos/demo_pairwise.md), [Sampling and Analysis](guides/sampling_and_analysis.md), and [Intent and Lifecycle](guides/intent_and_lifecycle.md)
- **Cassette workflows:** [Cassette Workspace Demo](demos/demo_cassette_workspace.md), [Cassette Workflow](guides/cassette_workflow.md), and [Cassette Solve Workflow](guides/cassette_solve_workflow.md)
- **Payload-Centric YIU Workflows:** [YIU Workspace Demo](demos/demo_yiu_workspace.md), [YIU Workflow](guides/yiu_workflow.md), [YIU Spec Reference](reference/yiu_spec.md), [YIU Artifacts](reference/yiu_artifacts.md), and [YIU Visual System](reference/yiu_visual_system.md)
- **Snapback workflows:** [Snapback Workflow](guides/snapback_workflow.md), [Released-product Snapback Workflow](guides/snapback_released_workflow.md), [Snapback Artifacts](reference/snapback_artifacts.md), [Released-product Snapback Artifacts](reference/released_snapback_artifacts.md), [Release-enzyme Catalogs](reference/release_enzyme_catalogs.md), and the checked-in [de033 released-product workspace README](../workspaces/de033/README.md)
- **Tracked study surfaces:** [Cruncher Study Status](operations/cruncher-study-status.md), [Cruncher Study Preflight](operations/cruncher-study-preflight.md), the checked-in [retron hairpin study status](../../../../docs/studies/retron_hairpin_design/status.md), the study [route map](../../../../docs/studies/retron_hairpin_design/routes.md), the [scar_nick package map](../src/scar_nick/README.md), and the repo-local [retron hairpin study skill](../../../../.agents/skills/retron-hairpin-study/SKILL.md)
- **Sample-backed YIU examples:** [demo_monotypic_tetr runbook](../workspaces/demo_monotypic_tetr/runbook.md) and [demo_monotypic_lexa runbook](../workspaces/demo_monotypic_lexa/runbook.md)
- **Study orchestration:** [Studies](guides/studies.md) and [Study Length vs Score](guides/study_length_vs_score.md)
- **Portfolio orchestration:** [Portfolio Aggregation](guides/portfolio_aggregation.md)
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

#### Payload-Centric YIU Workflows
- [YIU Workspace Demo](demos/demo_yiu_workspace.md)
- [YIU Workflow](guides/yiu_workflow.md)
- [YIU Spec Reference](reference/yiu_spec.md)
- [YIU Artifacts](reference/yiu_artifacts.md)
- [YIU Visual System](reference/yiu_visual_system.md)

#### Validate and Search Single-Nick Foldbacks
- [Snapback Workflow](guides/snapback_workflow.md)
- [Released-product Snapback Workflow](guides/snapback_released_workflow.md)
- [Snapback Artifacts](reference/snapback_artifacts.md)
- [Released-product Snapback Artifacts](reference/released_snapback_artifacts.md)
- [Release-enzyme Catalogs](reference/release_enzyme_catalogs.md)
- [de033 README](../workspaces/de033/README.md)
- [de033 runbook](../workspaces/de033/runbook.md)

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
- [Released-product Snapback Artifacts](reference/released_snapback_artifacts.md)
- [Release-enzyme Catalogs](reference/release_enzyme_catalogs.md)
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
