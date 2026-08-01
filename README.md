# ![dnadesign sequence toolkit](assets/dnadesign-banner.svg)

[![CI](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml)
[![Codecov](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg)](https://codecov.io/gh/e-south/dnadesign)
[![MIT license](https://img.shields.io/badge/license-MIT-3D8068.svg)](LICENSE)

`dnadesign` is a modular bioinformatics toolkit for designing sequence libraries, assembling DNA constructs, running sequence models, and analyzing the resulting datasets.

---

## Documentation

Use the docs index to choose a workflow, inspect existing work, or jump to a tool.

- [Docs index](docs/README.md): main index for workflow, tool, and repository docs.
- [Installation](docs/setup/installation.md): bootstrap the environment and run baseline verification commands.
- [Architecture](ARCHITECTURE.md), [Design](DESIGN.md), [Reliability](RELIABILITY.md), [Security](SECURITY.md), [Plans](PLANS.md), [Quality score](QUALITY_SCORE.md): repository-level contracts and governance records.
- [Developer docs](docs/dev/README.md): CI contracts, maintenance runbooks, and execution-planning references.

---

## Available tools

Pick the tool that matches the work you need to do. Each link opens the tool
README for commands, inputs, and outputs.

### Workspace and analysis tools

| Tool | Description | Coverage |
| --- | --- | --- |
| [**usr**](src/dnadesign/usr/README.md) | Store, inspect, validate, and sync tabular sequence records. | [![usr coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=usr)](https://codecov.io/gh/e-south/dnadesign?component=usr) |
| [**densegen**](src/dnadesign/densegen/README.md) | Generate DNA sequence libraries from declared design workspaces. | [![densegen coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=densegen)](https://codecov.io/gh/e-south/dnadesign?component=densegen) |
| [**infer**](src/dnadesign/infer/README.md) | Run sequence-model inference and write features back to datasets. | [![infer coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=infer)](https://codecov.io/gh/e-south/dnadesign?component=infer) |
| [**construct**](src/dnadesign/construct/README.md) | Build template-based constructs and declared sequence products. | [![construct coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=construct)](https://codecov.io/gh/e-south/dnadesign?component=construct) |
| [**opal**](src/dnadesign/opal/README.md) | Choose the next DNA designs from measured sequence results. | [![opal coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=opal)](https://codecov.io/gh/e-south/dnadesign?component=opal) |
| [**cluster**](src/dnadesign/cluster/README.md) | Group and visualize rows in numerical feature tables. | [![cluster coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=cluster)](https://codecov.io/gh/e-south/dnadesign?component=cluster) |
| [**latentdna**](src/dnadesign/latentdna/README.md) | Compare numerical sequence representations and publish review artifacts. | [![latentdna coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=latentdna)](https://codecov.io/gh/e-south/dnadesign?component=latentdna) |
| [**cruncher**](src/dnadesign/cruncher/README.md) | Solve constrained DNA sequence and assembly design tasks. | [![cruncher coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=cruncher)](https://codecov.io/gh/e-south/dnadesign?component=cruncher) |
| [**billboard**](src/dnadesign/billboard/README.md) | Measure how varied sequence motifs are within a library. | [![billboard coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=billboard)](https://codecov.io/gh/e-south/dnadesign?component=billboard) |
| [**libshuffle**](src/dnadesign/libshuffle/README.md) | Select representative subsets from large sequence libraries. | [![libshuffle coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=libshuffle)](https://codecov.io/gh/e-south/dnadesign?component=libshuffle) |
| [**nmf**](src/dnadesign/nmf/README.md) | Find recurring motif combinations with non-negative matrix factorization. | [![nmf coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=nmf)](https://codecov.io/gh/e-south/dnadesign?component=nmf) |
| [**permuter**](src/dnadesign/permuter/README.md) | Generate sequence variants and evaluate them with declared scoring tools. | [![permuter coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=permuter)](https://codecov.io/gh/e-south/dnadesign?component=permuter) |
| [**tfkdanalysis**](src/dnadesign/tfkdanalysis/README.md) | Summarize measured responses after transcription-factor knockdown. | [![tfkdanalysis coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=tfkdanalysis)](https://codecov.io/gh/e-south/dnadesign?component=tfkdanalysis) |
| [**aligner**](src/dnadesign/aligner/README.md) | Score pairwise sequence alignments with Biopython. | [![aligner coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=aligner)](https://codecov.io/gh/e-south/dnadesign?component=aligner) |
| [**thread**](src/dnadesign/thread/README.md) | Prepare protein-design requests and check predicted structures. | [![thread coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=thread)](https://codecov.io/gh/e-south/dnadesign?component=thread) |

### Artifact services

| Tool | Description | Coverage |
| --- | --- | --- |
| [**folding**](src/dnadesign/folding/README.md) | Predict and draw RNA secondary structures from sequence records. | [![folding coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=folding)](https://codecov.io/gh/e-south/dnadesign?component=folding) |
| [**baserender**](src/dnadesign/baserender/README.md) | Render sequence visuals from job files and visual contracts. | [![baserender coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=baserender)](https://codecov.io/gh/e-south/dnadesign?component=baserender) |

### Operator surfaces

| Tool | Description | Coverage |
| --- | --- | --- |
| [**ops**](src/dnadesign/ops/README.md) | Discover, run, and inspect jobs owned by repository tools. | [![ops coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=ops)](https://codecov.io/gh/e-south/dnadesign?component=ops) |
| [**notify**](src/dnadesign/notify/README.md) | Send webhook notifications for local runs and scheduler-backed jobs. | [![notify coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=notify)](https://codecov.io/gh/e-south/dnadesign?component=notify) |

### Shared contracts

| Tool | Description | Coverage |
| --- | --- | --- |
| [**contracts**](src/dnadesign/contracts/README.md) | Versioned schemas shared between tools. | [![contracts coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=contracts)](https://codecov.io/gh/e-south/dnadesign?component=contracts) |
---
