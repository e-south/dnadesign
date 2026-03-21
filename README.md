[![CI](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml/badge.svg)](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml) [![Codecov](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg)](https://codecov.io/gh/e-south/dnadesign) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

![dnadesign banner](assets/dnadesign-banner.svg)

`dnadesign` is a collection of modular bioinformatics tools for biological sequence design. It brings dataset management, sequence generation, construct realization, inference, clustering, optimization, and workflow orchestration into one repository.

---

## Documentation

Use the docs index to choose a workflow, inspect existing work, or jump to a tool.

- [Docs index](docs/README.md): main index for workflow, tool, and repository docs.
- [Installation](docs/installation.md): bootstrap the environment and run baseline verification commands.
- [Architecture](ARCHITECTURE.md), [Design](DESIGN.md), [Reliability](RELIABILITY.md), [Security](SECURITY.md), [Plans](PLANS.md), [Quality score](QUALITY_SCORE.md): repository-level contracts and governance records.
- [Developer docs](docs/dev/README.md): CI contracts, maintenance runbooks, and execution-planning references.

---

## Available tools

Use the tool READMEs below for task-specific docs and CLI entrypoints.

| Tool | Description | Coverage |
| --- | --- | --- |
| [**usr**](src/dnadesign/usr/README.md) | Inspect and manage USR datasets and Parquet-backed records. | [![usr coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=usr)](https://codecov.io/gh/e-south/dnadesign?component=usr) |
| [**ops**](src/dnadesign/ops/README.md) | Plan, submit, and inspect batch runbooks across tools. | [![ops coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=ops)](https://codecov.io/gh/e-south/dnadesign?component=ops) |
| [**notify**](src/dnadesign/notify/README.md) | Send webhook notifications for local runs and scheduler-backed jobs. | [![notify coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=notify)](https://codecov.io/gh/e-south/dnadesign?component=notify) |
| [**densegen**](src/dnadesign/densegen/README.md) | Generate DNA sequence libraries from workspace configs and inputs. CLI: `uv run dense`. | [![densegen coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=densegen)](https://codecov.io/gh/e-south/dnadesign?component=densegen) |
| [**infer**](src/dnadesign/infer/README.md) | Run sequence-model inference and write derived outputs back to datasets. | [![infer coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=infer)](https://codecov.io/gh/e-south/dnadesign?component=infer) |
| [**construct**](src/dnadesign/construct/README.md) | Build template-based DNA constructs and record placement metadata. | [![construct coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=construct)](https://codecov.io/gh/e-south/dnadesign?component=construct) |
| [**opal**](src/dnadesign/opal/README.md) | Run active-learning campaigns over labeled sequence datasets. | [![opal coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=opal)](https://codecov.io/gh/e-south/dnadesign?component=opal) |
| [**cluster**](src/dnadesign/cluster/README.md) | Cluster feature tables, render UMAPs, and write analysis outputs. | [![cluster coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=cluster)](https://codecov.io/gh/e-south/dnadesign?component=cluster) |
| [**billboard**](src/dnadesign/billboard/README.md) | Measure regulatory diversity across DenseGen libraries. | [![billboard coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=billboard)](https://codecov.io/gh/e-south/dnadesign?component=billboard) |
| [**libshuffle**](src/dnadesign/libshuffle/README.md) | Subsample libraries iteratively and analyze each round with `billboard`. | [![libshuffle coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=libshuffle)](https://codecov.io/gh/e-south/dnadesign?component=libshuffle) |
| [**nmf**](src/dnadesign/nmf/README.md) | Apply NMF to sequence libraries and summarize recurring TFBS combinations. | [![nmf coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=nmf)](https://codecov.io/gh/e-south/dnadesign?component=nmf) |
| [**latdna**](src/dnadesign/latdna/README.md) | Analyze DNA sequences in latent space. | [![latdna coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=latdna)](https://codecov.io/gh/e-south/dnadesign?component=latdna) |
| [**cruncher**](src/dnadesign/cruncher/README.md) | Optimize sequences against PWM-driven objectives. | [![cruncher coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=cruncher)](https://codecov.io/gh/e-south/dnadesign?component=cruncher) |
| [**tfkdanalysis**](src/dnadesign/tfkdanalysis/README.md) | Analyze TFKD libraries in PPTP-seq context. | [![tfkdanalysis coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=tfkdanalysis)](https://codecov.io/gh/e-south/dnadesign?component=tfkdanalysis) |
| [**aligner**](src/dnadesign/aligner/README.md) | Score global alignments with Biopython `PairwiseAligner`. | [![aligner coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=aligner)](https://codecov.io/gh/e-south/dnadesign?component=aligner) |
| [**baserender**](src/dnadesign/baserender/README.md) | Render sequences through a schema-driven adapter runtime. | [![baserender coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=baserender)](https://codecov.io/gh/e-south/dnadesign?component=baserender) |
| [**permuter**](src/dnadesign/permuter/README.md) | Generate sequence permutations and evaluate downstream outputs. | [![permuter coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=permuter)](https://codecov.io/gh/e-south/dnadesign?component=permuter) |
---
