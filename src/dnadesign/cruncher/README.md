![Cruncher banner](assets/cruncher-banner.svg)

Cruncher is a workspace-oriented DNA design tool with a few first-class workflow lanes:

- **`sample` and fixed-length optimization:** fetch TF evidence, lock exact inputs, parse, sample, analyze, then study or aggregate the resulting runs.
- **Cassette workflows:** validate or solve dual-context hairpin cassette specs and publish their own artifact and render contracts.
- **payload-centric YIU workflows:** start from either an exact `user_sequence` or a `sample_hit` resolved from public Sample outputs, search a 4 nt internal junction plus one or two per-position mismatches, and publish a deterministic three-view bundle under `outputs/<workflow>/`.

This README stays light on purpose. Use the [Docs map](docs/README.md) for the full index by workflow family, artifact type, and reference depth.

Use Cruncher when you need strict workspace contracts, deterministic artifacts, and clear boundaries between lanes. New families can be added beside these lanes without forcing `sample`, study, cassette, or YIU work into one artifact model.

`cruncher workspaces list` is the tool-local discovery surface for packaged
Cruncher workspaces and their machine runbooks. For repo-wide runbook discovery
across tools, start with [`docs/runbooks/README.md`](../../../docs/runbooks/README.md)
or `uv run ops catalog list --section tool-sources`.

### Start here

Start with the docs map, then pick the shortest path that matches your job:

1. [Docs map](docs/README.md): the comprehensive index.
2. [Sampling and analysis](docs/guides/sampling_and_analysis.md): the `sample` lane and the outputs that can feed YIU through `sample_hit`.
3. [YIU workflow](docs/guides/yiu_workflow.md): the `yiu` lane, from validation through bundle inspection.
4. [Cassette workspace demo](docs/demos/demo_cassette_workspace.md): the cassette lane end to end.
5. [CLI reference](docs/reference/cli.md): precise command contracts across all workflow families.
