![Cruncher banner](assets/cruncher-banner.svg)

Cruncher is the DNA design package in `dnadesign`. It groups six peer workflow families.
Registered family ids: `sample`, `cassette`, `yiu`, `snapback`, `study`, `portfolio`.

In practice that means workspace-based workflows for sequence optimization, cassette design, and YIU payload work, plus a Snapback lane and the study and portfolio flows that orchestrate those artifacts.

### Start here

1. [Docs map](docs/README.md): full routing by workflow family and doc type.
2. [Sampling and analysis](docs/guides/sampling_and_analysis.md): start here for the core optimization lane and its downstream study outputs.
3. [Cassette workspace demo](docs/demos/demo_cassette_workspace.md): run the cassette lane end to end from a packaged workspace.
4. [Released-product snapback workflow](docs/guides/snapback_released_workflow.md): follow the active released-product Snapback lane from probe through solve and bundle inspection.
5. [YIU workflow](docs/guides/yiu_workflow.md): follow the YIU lane from validation through bundle inspection.
6. [Studies and portfolio aggregation](docs/guides/studies.md): route into study and portfolio orchestration without collapsing them back into the `sample` family.
7. [CLI reference](docs/reference/cli.md): look up command contracts and flags across Cruncher.
