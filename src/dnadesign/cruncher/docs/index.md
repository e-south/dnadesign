## Cruncher docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-23

**Last updated by:** cruncher-maintainers on 2026-05-14

### Contents
- [Workflow families](#workflow-families)
- [Documentation map](#documentation-map)

### Workflow families

Cruncher currently registers seven peer workflow families.
Registered family ids: `sample`, `cassette`, `yiu`, `snapback`, `scar_nick`, `study`, `portfolio`.

- `sample` owns fixed-length optimization, artifact-only analysis, and export
- `cassette` owns cassette validation, design, solve, and show surfaces
- `yiu` owns payload-centric validation, render, and show surfaces
- `snapback` owns preserved-site and released-product single-nick foldback surfaces
- `scar_nick` owns retained-scar terminal-nick validation, design, and show surfaces
- `study` owns study orchestration over explicit source-family runs
- `portfolio` owns cross-study aggregation and reporting

The `study` and `portfolio` families orchestrate explicit source-family runs.
The currently shipped examples are sample-backed, but they are still separate
command families rather than hidden submodes of `sample`.
They aggregate explicit source-family outputs instead of redefining source
contracts.

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

#### Design Retained-Scar Terminal Nicks
- [Scar-Nick Workflow](guides/scar_nick_workflow.md)
- [scar_nick_teto runbook](../workspaces/scar_nick_teto/runbook.md)
- [scar_nick Package Map](../src/scar_nick/README.md)

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
