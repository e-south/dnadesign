![dnadesign sequence toolkit](https://raw.githubusercontent.com/e-south/dnadesign/v0.2.0/assets/dnadesign-banner.png)

[![CI](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml)
[![Codecov](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg)](https://codecov.io/gh/e-south/dnadesign)
[![MIT license](https://img.shields.io/badge/license-MIT-3D8068.svg)](https://github.com/e-south/dnadesign/blob/main/LICENSE)

`dnadesign` is a modular bioinformatics toolkit for designing sequence libraries,
assembling DNA constructs, running sequence models and analyzing their outputs.

## Get started

[Install the checkout](docs/setup/installation.md), then choose a
[tool](#available-tools) for its examples, inputs and outputs. The
[workflow index](docs/README.md) connects tools for tasks that span the monorepo.

For code changes, start with the [agent guide](AGENTS.md)
and the chosen tool's local instructions. They identify the owning code, runnable
examples and checks to use before broadening the work.

---

## Documentation

Use the docs index to choose a workflow, inspect existing work, or jump to a tool.

- [Docs index](https://github.com/e-south/dnadesign/blob/main/docs/README.md): main index for workflow, tool, and repository docs.
- [Installation](https://github.com/e-south/dnadesign/blob/main/docs/setup/installation.md): bootstrap the environment and run baseline verification commands.
- [Architecture](https://github.com/e-south/dnadesign/blob/main/ARCHITECTURE.md), [Design](https://github.com/e-south/dnadesign/blob/main/DESIGN.md), [Reliability](https://github.com/e-south/dnadesign/blob/main/RELIABILITY.md), [Security](https://github.com/e-south/dnadesign/blob/main/SECURITY.md), [Plans](https://github.com/e-south/dnadesign/blob/main/PLANS.md), [Quality score](https://github.com/e-south/dnadesign/blob/main/QUALITY_SCORE.md): repository-level contracts and governance records.
- [Developer docs](https://github.com/e-south/dnadesign/blob/main/docs/dev/README.md): CI contracts, maintenance runbooks, and execution-planning references.

---

## Available tools

Pick the tool that matches the work you need to do. Each link opens the tool
README for commands, inputs, and outputs.

### Design, modeling, and analysis

| Tool | Description |
| --- | --- |
| [**usr**](src/dnadesign/usr/) | Store, check, and sync tables of sequence records. |
| [**densegen**](src/dnadesign/densegen/) | Generate DNA sequence libraries from saved design inputs. |
| [**infer**](src/dnadesign/infer/) | Run sequence models and add their predictions to datasets. |
| [**construct**](src/dnadesign/construct/) | Build DNA constructs from reusable templates. |
| [**junction**](src/dnadesign/junction/) | Turn exact DNA targets into checked three-way-junction oligo plans. |
| [**opal**](src/dnadesign/opal/) | Use measured results to choose which DNA designs to test next. |
| [**cluster**](src/dnadesign/cluster/) | Group similar rows and visualize the groups. |
| [**latentdna**](src/dnadesign/latentdna/) | Compare sequence features and save tables and plots for review. |
| [**cruncher**](src/dnadesign/cruncher/) | Design DNA sequences and assembly plans within stated limits. |
| [**billboard**](src/dnadesign/billboard/) | Measure how much binding-site content varies across a sequence library. |
| [**libshuffle**](src/dnadesign/libshuffle/) | Choose smaller, representative sets from large sequence libraries. |
| [**nmf**](src/dnadesign/nmf/) | Find motif patterns that repeatedly occur together in a data table. |
| [**permuter**](src/dnadesign/permuter/) | Create sequence variants and evaluate them with selected tools. |
| [**tfkdanalysis**](src/dnadesign/tfkdanalysis/) | Summarize experiments that reduce selected transcription factors. |
| [**aligner**](src/dnadesign/aligner/) | Compare pairs or sets of DNA and protein sequences. |
| [**thread**](src/dnadesign/thread/) | Prepare protein-design jobs and check the predicted structures. |

### Draw and inspect results

| Tool | Description |
| --- | --- |
| [**folding**](src/dnadesign/folding/) | Predict and draw how RNA sequences fold. |
| [**baserender**](src/dnadesign/baserender/) | Draw annotated sequence maps from saved jobs. |

### Run and monitor jobs

| Tool | Description |
| --- | --- |
| [**ops**](src/dnadesign/ops/) | Find, run, and inspect jobs across dnadesign. |
| [**notify**](src/dnadesign/notify/) | Send notifications for local and scheduled jobs. |

### Data shared between tools

| Tool | Description |
| --- | --- |
| [**contracts**](src/dnadesign/contracts/) | Define versioned file formats shared between tools. |
---
