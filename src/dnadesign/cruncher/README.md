![Cruncher banner](assets/cruncher-banner.svg)

Cruncher is a workspace-oriented DNA design tool with three first-class workflow families today:

- **Fixed-length optimization workspaces:** fetch TF evidence, lock exact inputs, run Gibbs annealing MCMC with MMR elite selection, then analyze, study, and aggregate the resulting runs.
- **Cassette workspaces:** validate or solve dual-context hairpin cassette specs, publish explicit artifacts and shared visual contracts, and hand off render jobs without reusing the `sample` run layout.
- **payload-centric YIU workflows:** take either an exact `user_sequence` or a `sample_hit` resolved from public Sample outputs, search a 4 nt internal junction plus one or two per-position mismatches, optionally score those candidates against PWM context, and publish a deterministic three-view bundle under `outputs/<workflow>/`.

Use Cruncher when you need strict workspace contracts, deterministic artifacts, and a clear separation between workflow families. New families can be added beside these lanes without forcing sample workspaces, study workspaces, cassette workspaces, or YIU workspaces into one artifact model.

`cruncher workspaces list` is the tool-local discovery surface for packaged
Cruncher workspaces and their machine runbooks. For repo-wide runbook discovery
across tools, start with [`docs/runbooks/README.md`](../../../docs/runbooks/README.md)
or `uv run ops catalog list --section tool-sources`.

### Start here

Use the docs map for the full index, then pick the shortest path that matches your job:

1. [Docs map](docs/README.md): route by workflow family, artifact type, and reference depth.
2. [Pairwise demo](docs/demos/demo_pairwise.md): run the core fixed-length optimization lane end to end.
3. [MultiTF demo](docs/demos/demo_multitf.md): extend the same lane to a three-TF workspace.
4. [Project workspace demo](docs/demos/project_all_tfs.md): run the larger study-ready workspace flow.
5. [Cassette workspace demo](docs/demos/demo_cassette_workspace.md): scaffold a cassette workspace, solve, and render QA outputs in place.
6. [YIU workspace demo](docs/demos/demo_yiu_workspace.md): run the checked-in user-sequence YIU workspace and inspect the published payload bundle.
7. [YIU workflow](docs/guides/yiu_workflow.md): follow the `init-workspace -> validate -> render -> show` flow, including `sample_hit` handoff from Sample outputs.
8. [YIU spec reference](docs/reference/yiu_spec.md): author strict `.yiu.yaml` specs, including `sample_hit` sources and PWM options.
9. [YIU artifacts](docs/reference/yiu_artifacts.md): inspect the bundle files, render status, and `show` surface.
10. [YIU visual system](docs/reference/yiu_visual_system.md): understand how the three YIU views are organized.
11. [Sampling and analysis](docs/guides/sampling_and_analysis.md): understand the `sample` lane that can feed `sample_hit` inputs into YIU.
12. [Studies guide](docs/guides/studies.md): run workspace-scoped sweeps over the fixed-length optimization lane.
13. [Portfolio aggregation](docs/guides/portfolio_aggregation.md): aggregate selected runs across workspaces.
14. [CLI reference](docs/reference/cli.md): precise command contracts across all workflow families.
