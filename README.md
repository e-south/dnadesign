![dnadesign sequence toolkit](assets/dnadesign-banner.svg)

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
| [**infer**](src/dnadesign/infer/README.md) | Run sequence models and add their predictions to datasets. |
| [**construct**](src/dnadesign/construct/README.md) | Build DNA constructs from reusable templates. |
| [**msd**](src/dnadesign/msd/README.md) | Resolve and compile typed Retron MSD designs. |
| [**junction**](src/dnadesign/junction/README.md) | Turn exact DNA targets into checked three-way-junction oligo plans. |
| [**opal**](src/dnadesign/opal/README.md) | Use measured results to choose which DNA designs to test next. |
| [**cluster**](src/dnadesign/cluster/README.md) | Group similar rows and visualize the groups. |
| [**latentdna**](src/dnadesign/latentdna/README.md) | Compare sequence features and save tables and plots for review. |
| [**cruncher**](src/dnadesign/cruncher/README.md) | Design DNA sequences and assembly plans within stated limits. |
| [**billboard**](src/dnadesign/billboard/README.md) | Measure how much binding-site content varies across a sequence library. |
| [**libshuffle**](src/dnadesign/libshuffle/README.md) | Choose smaller, representative sets from large sequence libraries. |
| [**nmf**](src/dnadesign/nmf/README.md) | Find motif patterns that repeatedly occur together in a data table. |
| [**permuter**](src/dnadesign/permuter/README.md) | Create sequence variants and evaluate them with selected tools. |
| [**tfkdanalysis**](src/dnadesign/tfkdanalysis/README.md) | Summarize experiments that reduce selected transcription factors. |
| [**aligner**](src/dnadesign/aligner/README.md) | Compare pairs or sets of DNA and protein sequences. |
| [**thread**](src/dnadesign/thread/README.md) | Prepare protein-design jobs and check the predicted structures. |

### Draw and inspect results

| Tool | Description |
| --- | --- |
| [**folding**](src/dnadesign/folding/README.md) | Predict and draw how RNA sequences fold. |
| [**baserender**](src/dnadesign/baserender/README.md) | Draw annotated sequence maps from saved jobs. |

### Run and monitor jobs

| Tool | Description |
| --- | --- |
| [**ops**](src/dnadesign/ops/README.md) | Find, run, and inspect jobs across dnadesign. |
| [**notify**](src/dnadesign/notify/README.md) | Send notifications for local and scheduled jobs. |

### Data shared between tools

| Tool | Description |
| --- | --- |
| [**contracts**](src/dnadesign/contracts/README.md) | Define versioned file formats shared between tools. |
---
