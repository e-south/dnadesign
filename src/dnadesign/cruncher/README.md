---
doc_id: cruncher-package
title: Cruncher
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
---

![Cruncher banner](assets/cruncher-banner.svg)

Cruncher runs reproducible DNA design jobs. It can optimize fixed-length
sequences against motif models, search multi-part cassette designs, assess
declared foldback and nick geometries, inspect payload windows, and aggregate
prior run artifacts. Each route validates its own workspace and writes a typed
bundle. Exact command IDs belong in the CLI and architecture references.

## Documentation

1. [Choose a workflow](docs/README.md): route by the design job you need.
2. [Optimize fixed-length sequences](docs/guides/sampling_and_analysis.md): score and sample sequences against declared motif models.
3. [Design multi-part cassettes](docs/demos/demo_cassette_workspace.md): validate and search one cassette workspace end to end.
4. [Check a released foldback design](docs/guides/snapback_released_workflow.md): inspect the precursor, release site, and expected product geometry.
5. [Check a terminal nick design](docs/guides/scar_nick_workflow.md): validate a retained-scar nick geometry.
6. [Inspect a payload window](docs/guides/yiu_workflow.md): validate and render one declared payload design.
7. [Aggregate parameter sweeps](docs/guides/studies.md): summarize explicit prior runs without changing their source contracts.
8. [CLI reference](docs/reference/cli.md): look up exact command names and flags.
