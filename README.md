![dnadesign sequence toolkit](https://raw.githubusercontent.com/e-south/dnadesign/v0.3.0rc1/assets/dnadesign-banner.png)

[![CI](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml)
[![Codecov](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg)](https://codecov.io/gh/e-south/dnadesign)
[![MIT license](https://img.shields.io/badge/license-MIT-3D8068.svg)](https://github.com/e-south/dnadesign/blob/main/LICENSE)

`dnadesign` is a modular bioinformatics toolkit for designing sequence libraries,
assembling DNA constructs, running sequence models and analyzing their outputs.

## Get started

[Choose an installation](docs/setup/installation.md) for the scoring API, OPAL
or the full toolkit, then choose a
[tool](#available-tools) for its examples, inputs and outputs. The
[workflow index](docs/README.md) connects tools for tasks that span the monorepo.

## Available tools

Pick the tool that matches the work you need to do. Each link opens the tool
README for commands, inputs, and outputs.

### Design, modeling, and analysis

| Tool | Description |
| --- | --- |
| [**seqledger**](src/dnadesign/seqledger/) | Store, check, and sync tables of sequence records. |
| [**infer**](src/dnadesign/infer/) | Run sequence models and add their predictions to datasets. |
| [**densegen**](src/dnadesign/densegen/) | Generate sequence libraries from supplied parts and packing constraints. |
| [**construct**](src/dnadesign/construct/) | Build DNA constructs from reusable templates. |
| [**junction**](src/dnadesign/junction/) | Turn exact DNA targets into checked three-way-junction oligo plans. |
| [**opal**](src/dnadesign/opal/) | Use measured results to choose which DNA designs to test next. |
| [**cluster**](src/dnadesign/cluster/) | Group similar rows and visualize the groups. |
| [**latentdna**](src/dnadesign/latentdna/) | Compare sequence features and save tables and plots for review. |
| [**cruncher**](src/dnadesign/cruncher/) | Design DNA sequences and assembly plans within stated limits. |
| [**permuter**](src/dnadesign/permuter/) | Create sequence variants and evaluate them with selected tools. |
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

## Contributing and reference

- [Docs index](docs/README.md): workflow and tool documentation.
- [Architecture](ARCHITECTURE.md), [Design](DESIGN.md) and [Reliability](RELIABILITY.md): boundaries and operating principles.
- [Developer docs](docs/dev/README.md): contribute changes and run validation checks.
- [Security](SECURITY.md), [Plans](PLANS.md) and [Quality score](QUALITY_SCORE.md): security policy and maintenance.
