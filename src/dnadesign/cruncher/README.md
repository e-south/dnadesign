![Cruncher banner](assets/cruncher-banner.svg)

Cruncher solves bounded DNA design problems through explicit workflow
families. Each family owns its command route, workspace contract, and artifact
tree so studies can combine outputs without one monolithic run shape.
Registered family ids: `sample`, `cassette`, `yiu`, `snapback`, `scar_nick`, `study`, `portfolio`.

Use Cruncher when you need to solve for primitive design parts: fixed-length
sequences, cassettes, YIU payload windows, Snapback foldbacks, or scar-nick
retained-scar junctions. Study and portfolio flows orchestrate those outputs;
they do not replace the primitive routes.

### Start here

1. [Docs map](docs/README.md): full routing by workflow family and doc type.
2. [Sampling and analysis](docs/guides/sampling_and_analysis.md): run the fixed-length optimization lane.
3. [Cassette workspace demo](docs/demos/demo_cassette_workspace.md): run cassette design end to end.
4. [Released-product snapback workflow](docs/guides/snapback_released_workflow.md): solve and inspect released-product Snapback bundles.
5. [Scar-nick workflow](docs/guides/scar_nick_workflow.md): validate and design retained-scar terminal-nick panels.
6. [YIU workflow](docs/guides/yiu_workflow.md): validate, render, and inspect payload-centric YIU bundles.
7. [Studies and portfolio aggregation](docs/guides/studies.md): orchestrate explicit source-family artifacts.
8. [CLI reference](docs/reference/cli.md): look up command contracts and flags across Cruncher.
