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

### Design, modeling, and analysis

| Tool | Description |
| --- | --- |
| [**usr**](src/dnadesign/usr/README.md) | Store, check, and sync tables of sequence records. |
| [**densegen**](src/dnadesign/densegen/README.md) | Generate DNA sequence libraries from saved design inputs. |
| [**infer**](src/dnadesign/infer/README.md) | Run sequence models and add their results to datasets. |
| [**construct**](src/dnadesign/construct/README.md) | Build DNA constructs from reusable templates. |
| [**opal**](src/dnadesign/opal/README.md) | Choose the next DNA designs from measured results. |
| [**cluster**](src/dnadesign/cluster/README.md) | Group and visualize rows in numerical data tables. |
| [**latentdna**](src/dnadesign/latentdna/README.md) | Compare numerical sequence features and save review-ready results. |
| [**cruncher**](src/dnadesign/cruncher/README.md) | Find DNA sequences and assembly plans that satisfy stated constraints. |
| [**billboard**](src/dnadesign/billboard/README.md) | Measure how varied sequence motifs are within a library. |
| [**libshuffle**](src/dnadesign/libshuffle/README.md) | Select representative subsets from large sequence libraries. |
| [**nmf**](src/dnadesign/nmf/README.md) | Find recurring groups of sequence motifs in data tables. |
| [**permuter**](src/dnadesign/permuter/README.md) | Generate sequence variants and score them with chosen tools. |
| [**tfkdanalysis**](src/dnadesign/tfkdanalysis/README.md) | Summarize measured responses after transcription-factor knockdown. |
| [**aligner**](src/dnadesign/aligner/README.md) | Compare pairs or sets of DNA and protein sequences. |
| [**thread**](src/dnadesign/thread/README.md) | Prepare requests for protein-design models and check predicted structures. |

### Draw and inspect results

| Tool | Description |
| --- | --- |
| [**folding**](src/dnadesign/folding/README.md) | Predict and draw RNA secondary structures from sequence records. |
| [**baserender**](src/dnadesign/baserender/README.md) | Draw annotated sequence diagrams from saved job files. |

### Run and monitor jobs

| Tool | Description |
| --- | --- |
| [**ops**](src/dnadesign/ops/README.md) | Find, run, and inspect jobs owned by repository tools. |
| [**notify**](src/dnadesign/notify/README.md) | Send webhook notifications for local and scheduled jobs. |

### Data shared between tools

| Tool | Description |
| --- | --- |
| [**contracts**](src/dnadesign/contracts/README.md) | Define versioned data formats shared between tools. |
---
