## Cruncher Documentation Index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-05


**Last updated by:** cruncher-maintainers on 2026-04-05

Cruncher currently has three first-class workflow families:

- fixed-length PWM optimization workspaces built around `fetch -> lock -> parse -> sample -> analyze`
- cassette workspaces built around `cassette init-workspace|validate|design|solve|show`
- payload-centric YIU v4 workspaces built around `yiu init-workspace|validate|render|show`

Studies and portfolios build on the fixed-length optimization lane. Cassette and YIU runs keep their own workspace and artifact contracts. Future lanes should sit beside these families rather than overload them.

For YIU, the shortest mental model is:

- start from either `user_sequence` or `sample_hit`
- validate and normalize the payload
- search valid 4 nt internal junction and mismatch plans
- publish one deterministic bundle of three views
- optionally render `payload_views.pdf`

### Quick routes

- Need the checked-in user-sequence demo: [YIU Workspace Demo](demos/demo_yiu_workspace.md)
- Need a sample-backed YIU example that starts from `sample` outputs:
  [demo_monotypic_tetr runbook](../workspaces/demo_monotypic_tetr/runbook.md) and
  [demo_monotypic_lexa runbook](../workspaces/demo_monotypic_lexa/runbook.md)
- Need the public YIU command flow: [YIU Workflow](guides/yiu_workflow.md)
- Need the strict input contract: [YIU Spec Reference](reference/yiu_spec.md)
- Need emitted files and `show` semantics: [YIU Artifacts](reference/yiu_artifacts.md)
- Need render hierarchy and view emphasis: [YIU Visual System](reference/yiu_visual_system.md)
- Need the upstream Sample lane: [Sampling and Analysis](guides/sampling_and_analysis.md)

### Progressive disclosure route
1. Choose one workflow family under **Choose a workflow family**.
2. Start with one demo or guide from that family.
3. Use **Reference contracts** when you need strict CLI, schema, or artifact behavior.

<!-- docs:toc:off -->

### Choose a workflow family

- **Fixed-length optimization workspaces:** start with [Pairwise Demo](demos/demo_pairwise.md), then move to [Sampling and Analysis](guides/sampling_and_analysis.md) and [Intent and Lifecycle](guides/intent_and_lifecycle.md).
- **Cassette workspaces:** start with [Cassette Workspace Demo](demos/demo_cassette_workspace.md), then use [Cassette Workflow](guides/cassette_workflow.md) or [Cassette Solve Workflow](guides/cassette_solve_workflow.md).
- **Payload-Centric YIU Workflows:** start with [YIU Workspace Demo](demos/demo_yiu_workspace.md), then use [YIU Workflow](guides/yiu_workflow.md), [YIU Spec Reference](reference/yiu_spec.md), [YIU Artifacts](reference/yiu_artifacts.md), and [YIU Visual System](reference/yiu_visual_system.md). Public CLI: `yiu init-workspace|validate|render|show` with `split_yiu_payload_rendering_v4`.
  Use the demo for the checked-in workspace, the workflow guide for command flow, the spec reference for schema and normalization, the artifacts page for emitted files and `show`, and the visual-system page for view hierarchy.
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
