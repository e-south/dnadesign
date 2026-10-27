![Cruncher banner](assets/cruncher-banner.svg)

Cruncher runs reproducible DNA design jobs. It can optimize fixed-length
sequences against motif models, search multi-part cassette designs, inspect payload windows, and aggregate
prior run artifacts. Each route validates its own workspace and writes a typed
bundle. Exact command IDs belong in the CLI and architecture references.

## Documentation

1. [Choose a workflow](docs/README.md): route by the design job you need.
2. [Optimize fixed-length sequences](docs/guides/sampling_and_analysis.md): score and sample sequences against declared motif models.
3. [Design multi-part cassettes](docs/demos/demo_cassette_workspace.md): validate and search one cassette workspace end to end.
4. [Inspect a payload window](docs/guides/yiu_workflow.md): validate and render one declared payload design.
5. [Aggregate parameter sweeps](docs/guides/studies.md): summarize explicit prior runs without changing their source contracts.
6. [CLI reference](docs/reference/cli.md): look up exact command names and flags.
