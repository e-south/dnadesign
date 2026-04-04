![Cruncher banner](assets/cruncher-banner.svg)

Cruncher is a workspace-oriented DNA design tool with three first-class workflow families today:

- **Fixed-length optimization workspaces:** fetch TF evidence, lock exact inputs, run Gibbs annealing MCMC with MMR elite selection, then analyze, study, and aggregate the resulting runs.
- **Cassette workspaces:** validate or solve dual-context hairpin cassette specs, publish explicit artifacts and shared visual contracts, and hand off render jobs without reusing the `sample` run layout.
- **payload-centric YIU workflows:** validate strict `.yiu.yaml` payload specs, normalize `user_sequence` or `sample_hit` inputs, exhaustively optimize a 4-nt junction and per-position mismatches with optional PWM context, and publish three canonical BaseRender-ready views under `outputs/<workflow>/`.

Use Cruncher when you need strict workspace contracts, deterministic artifacts, and a clear separation between workflow families. New families can be added beside these lanes without forcing sample workspaces, study workspaces, cassette workspaces, or YIU workspaces into one artifact model.

`cruncher workspaces list` is the tool-local discovery surface for packaged
Cruncher workspaces and their machine runbooks. For repo-wide runbook discovery
across tools, start with [`docs/runbooks/README.md`](../../../docs/runbooks/README.md)
or `uv run ops catalog list --section tool-sources`.

### Start here

Use the docs map for the full index, then pick the shortest workflow that matches your job:

1. [Docs map](docs/README.md): route by workflow family, artifact type, and reference depth.
2. [Pairwise demo](docs/demos/demo_pairwise.md): run the core fixed-length optimization lane end to end.
3. [MultiTF demo](docs/demos/demo_multitf.md): extend the same lane to a three-TF workspace.
4. [Project workspace demo](docs/demos/project_all_tfs.md): run the larger study-ready workspace flow.
5. [Cassette workspace demo](docs/demos/demo_cassette_workspace.md): scaffold a cassette workspace, solve, and render QA outputs in place.
6. [YIU workspace demo](docs/demos/demo_yiu_workspace.md): scaffold a YIU workspace and inspect the published payload bundle.
7. [Sampling and analysis](docs/guides/sampling_and_analysis.md): understand `sample`, Gibbs annealing, and analysis outputs.
8. [Cassette workflow](docs/guides/cassette_workflow.md): validate and materialize an authored cassette spec.
9. [YIU workflow](docs/guides/yiu_workflow.md): validate and render a payload-centric YIU v4 spec.
10. [Studies guide](docs/guides/studies.md): run workspace-scoped sweeps over the fixed-length optimization lane.
11. [Portfolio aggregation](docs/guides/portfolio_aggregation.md): aggregate selected runs across workspaces.
12. [CLI reference](docs/reference/cli.md): precise command contracts across all workflow families.
