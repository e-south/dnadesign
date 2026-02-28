![Cruncher banner](assets/cruncher-banner.svg)

Cruncher designs short, fixed-length DNA sequences that score highly across one or more transcription factor (TF) position weight matrices (PWMs), using Gibbs annealing and transcription-factor-binding-site core maximal marginal relevance (MMR) to select a diverse elite set. Use the demos for end-to-end workflows and the references for precise schema/command-line behavior.

See the [repository docs index](../../../docs/README.md) for cross-tool workflow routing.

Interface contracts are layered:

- Repository-level contracts: [Architecture](../../../ARCHITECTURE.md), [Design](../../../DESIGN.md), [Reliability](../../../RELIABILITY.md), [Security](../../../SECURITY.md)
- Tool-level contracts: [Config reference](docs/reference/config.md), [CLI reference](docs/reference/cli.md), [Artifacts reference](docs/reference/artifacts.md), [Architecture reference](docs/reference/architecture.md)
- Workflow entrypoints: [Demos](docs/README.md#documentation-by-workflow) and then targeted guides

- **What it is:** an optimization engine for designing **short, fixed-length DNA** sequences that jointly satisfy one or more transcription factor position weight matrices, then returning a **diverse elite set**.
- **When to use:** multi-transcription-factor promoter/operator design under tight length constraints; motif-compatibility tradeoff exploration; producing a small, diverse candidate set for assays; workspace-scoped parameter studies with aggregate comparison.
- **Mental model:** deterministic data prep (`fetch`/`lock`) + strict Gibbs annealing optimization (`sample`) + artifact-native analytics (`analyze`).

### Documentation map

Use the demos as the default entrypoint:

1. [Docs map](docs/README.md): route to demos, guides, and references by workflow.
2. [Two-transcription-factor demo](docs/demos/demo_pairwise.md): run the pairwise optimization path end to end.
3. [Three-transcription-factor demo](docs/demos/demo_multitf.md): run a multitf optimization workflow with full outputs.
4. [Project workspace (all TFs)](docs/demos/project_all_tfs.md): execute the full-project workspace flow across all TFs.
5. [Intent and lifecycle](docs/guides/intent_and_lifecycle.md): understand stage order and artifact expectations.
6. [Sampling and analysis](docs/guides/sampling_and_analysis.md): tune sampling runs and post-run analysis.
7. [Ingestion guide](docs/guides/ingestion.md): prepare and validate upstream motif inputs.
8. [MEME Suite guide](docs/guides/meme_suite.md): run MEME/FIMO integration paths and expected artifacts.
9. [Studies guide](docs/guides/studies.md): orchestrate repeatable study runs.
10. [Portfolio aggregation](docs/guides/portfolio_aggregation.md): combine study outputs into portfolio-level summaries.
11. [Artifacts reference](docs/reference/artifacts.md): inspect output paths and file contracts.
12. [Config reference](docs/reference/config.md): map configuration fields to runtime behavior.
13. [CLI reference](docs/reference/cli.md): command contracts, flags, and operational expectations.
14. [Architecture reference](docs/reference/architecture.md): module boundaries and dataflow.
