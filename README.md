[![CI](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml/badge.svg)](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml) [![Codecov](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg)](https://codecov.io/gh/e-south/dnadesign)

![dnadesign banner](assets/dnadesign-banner.svg)

`dnadesign` is a collection of modular bioinformatic pipelines and helper packages for biological sequence design. The repository is organized as interoperable tools so generation, realization, inference, clustering, optimization, and workflow orchestration can stay explicit at their boundaries instead of collapsing into one opaque pipeline.

---

## New here?

```bash
uv sync --locked --group dev
uv run ops catalog list --simple
uv run ops progress explain usr.data-plane.promoter-feature-matrix
```

- If `uv` is not installed or the environment is not usable yet, start with [Installation](docs/installation.md). Once `uv run <tool> --help` works, return here or go straight to `uv run ops catalog list --simple`.

| Start here if you want to... | Use this first | What you get next |
| --- | --- | --- |
| Understand the repository from the shell | `uv run ops catalog list --simple` | A task-first view of registered workflows and tool docs before you need taxonomy. |
| Inspect one registered route in detail | `uv run ops catalog show <registry-id>` | Owner docs, related routes, required status inputs, and next commands. |
| Understand what a status command needs | `uv run ops progress explain <registry-id>` | Required flags, ready-to-paste `progress show` command, and notes for special cases like OPAL config resolution. |
| Browse the full docs map | [Docs index](docs/README.md) | Workflow routes, shell routes, tool docs, and repository policy. |

- To orchestrate or inspect batch workflows: start with [Ops](src/dnadesign/ops/README.md) or run `uv run ops catalog list --simple`.
- DenseGen lives under `densegen`, but the CLI command is `uv run dense --help`.

---

## Documentation

Use the docs index to choose the next deep procedure by ownership plane, find the next concrete workflow or tool doc, then drop into the owning package for operational detail.

- [Docs index](docs/README.md): central route map for workflow and tool documentation.
- [Installation](docs/installation.md): bootstrap the environment and run baseline verification commands.
- [Architecture](ARCHITECTURE.md), [Design](DESIGN.md), [Reliability](RELIABILITY.md), [Security](SECURITY.md), [Plans](PLANS.md), [Quality score](QUALITY_SCORE.md): repository-level contracts and governance records.
- [Developer docs](docs/dev/README.md): CI contracts, maintenance runbooks, and execution-planning references.

---

## Available tools

Package names and CLI command names are usually aligned, but not always. DenseGen is the main exception: the package/docs name is `densegen`, while the CLI entrypoint is `dense`.

| Tool | Description | Coverage |
| --- | --- | --- |
| [**usr**](src/dnadesign/usr/README.md) | Universal Sequence Record utilities for inspecting datasets and Parquet files. | [![usr coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=usr)](https://codecov.io/gh/e-south/dnadesign?component=usr) |
| [**ops**](src/dnadesign/ops/README.md) | Runbook-driven orchestration for deterministic batch workflows across tools. | [![ops coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=ops)](https://codecov.io/gh/e-south/dnadesign?component=ops) |
| [**notify**](src/dnadesign/notify/README.md) | Tool-agnostic webhook notifier for batch runs (Slack, Discord, generic webhooks). | [![notify coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=notify)](https://codecov.io/gh/e-south/dnadesign?component=notify) |
| [**densegen**](src/dnadesign/densegen/README.md) | DNA sequence design pipeline built on the [`dense-arrays`](https://github.com/e-south/dense-arrays) framework. | [![densegen coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=densegen)](https://codecov.io/gh/e-south/dnadesign?component=densegen) |
| [**infer**](src/dnadesign/infer/README.md) | Model-agnostic wrapper for DNA/protein language models such as [Evo2](https://github.com/ArcInstitute/evo2/tree/main). | [![infer coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=infer)](https://codecov.io/gh/e-south/dnadesign?component=infer) |
| [**construct**](src/dnadesign/construct/README.md) | Builds larger DNA contexts by placing one or more parts into an explicit template to realize regions, plasmids, and related constructs. | [![construct coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=construct)](https://codecov.io/gh/e-south/dnadesign?component=construct) |
| [**opal**](src/dnadesign/opal/README.md) | [EVOLVEpro-style](https://www.science.org/doi/10.1126/science.adr6006) active-learning tool for DNA/protein sequence design campaigns. | [![opal coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=opal)](https://codecov.io/gh/e-south/dnadesign?component=opal) |
| [**cluster**](src/dnadesign/cluster/README.md) | Parquet/CSV-first toolkit for clustering, UMAP visualization, and related analyses. | [![cluster coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=cluster)](https://codecov.io/gh/e-south/dnadesign?component=cluster) |
| [**billboard**](src/dnadesign/billboard/README.md) | Quantifies regulatory diversity of dense-array DNA libraries generated by `densegen`. | [![billboard coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=billboard)](https://codecov.io/gh/e-south/dnadesign?component=billboard) |
| [**libshuffle**](src/dnadesign/libshuffle/README.md) | Iterative subsampling workflow that uses `billboard` as its analysis engine. | [![libshuffle coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=libshuffle)](https://codecov.io/gh/e-south/dnadesign?component=libshuffle) |
| [**nmf**](src/dnadesign/nmf/README.md) | Applies NMF to sequence libraries to identify higher-order TFBS combinations. | [![nmf coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=nmf)](https://codecov.io/gh/e-south/dnadesign?component=nmf) |
| [**latdna**](src/dnadesign/latdna/README.md) | Pipeline for latent-space analysis of DNA sequences. | [![latdna coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=latdna)](https://codecov.io/gh/e-south/dnadesign?component=latdna) |
| [**cruncher**](src/dnadesign/cruncher/README.md) | PWM-driven sequence optimization pipeline with pluggable parsers and optimizers. | [![cruncher coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=cruncher)](https://codecov.io/gh/e-south/dnadesign?component=cruncher) |
| [**tfkdanalysis**](src/dnadesign/tfkdanalysis/README.md) | Pipeline for TFKD analysis with PPTP-seq context ([Han et al., 2023](https://doi.org/10.1038/s41467-023-41572-4)). | [![tfkdanalysis coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=tfkdanalysis)](https://codecov.io/gh/e-south/dnadesign?component=tfkdanalysis) |
| [**aligner**](src/dnadesign/aligner/README.md) | Wrapper around Biopython `PairwiseAligner` for Needleman-Wunsch-style global alignment scoring. | [![aligner coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=aligner)](https://codecov.io/gh/e-south/dnadesign?component=aligner) |
| [**baserender**](src/dnadesign/baserender/README.md) | Contract-first sequence rendering runtime with strict schemas and adapter-based integration. | [![baserender coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=baserender)](https://codecov.io/gh/e-south/dnadesign?component=baserender) |
| [**permuter**](src/dnadesign/permuter/README.md) | Pipeline for biological sequence permutation and downstream evaluation. | [![permuter coverage](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg?component=permuter)](https://codecov.io/gh/e-south/dnadesign?component=permuter) |
---
