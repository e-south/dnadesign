![Cruncher banner](assets/cruncher-banner.svg)

Cruncher is a workspace-based DNA design tool with three workflow families:

- **`sample` and fixed-length optimization:** build fixed-length sequence runs, then analyze, study, or aggregate them.
- **Cassette workflows:** validate, design, and search cassette workspaces.
- **payload-centric YIU workflows:** validate, render, and inspect payload-splitting workspaces.

Studies and portfolio aggregation build on the fixed-length optimization lane. Use the [Docs map](docs/README.md) for demos, workflow guides, and reference pages.

`cruncher workspaces list` is the tool-local discovery surface for packaged
Cruncher workspaces and their machine runbooks. For repo-wide runbook discovery
across tools, start with [`docs/runbooks/README.md`](../../../docs/runbooks/README.md)
or `uv run ops catalog list --section tool-sources`.

### Start here

Start with the docs map, then pick the shortest path that matches your job:

1. [Docs map](docs/README.md): full routing by workflow family and doc type.
2. [Sampling and analysis](docs/guides/sampling_and_analysis.md): the core fixed-length optimization lane and its analysis outputs.
3. [Cassette workspace demo](docs/demos/demo_cassette_workspace.md): the cassette lane end to end.
4. [YIU workflow](docs/guides/yiu_workflow.md): the payload workflow from validation through bundle inspection.
5. [CLI reference](docs/reference/cli.md): command contracts across workflow families.
