## Cruncher

![Cruncher banner](assets/cruncher-banner.svg)

Cruncher designs short, fixed-length DNA sequences that score highly across one or more transcription factor (TF) position weight matrices (PWMs), using Gibbs annealing and transcription-factor-binding-site core maximal marginal relevance (MMR) to select a diverse elite set. Use the demos for end-to-end workflows and the references for precise schema/command-line behavior.

- **What it is:** an optimization engine for designing **short, fixed-length DNA** sequences that jointly satisfy one or more transcription factor position weight matrices, then returning a **diverse elite set**.
- **When to use:** multi-transcription-factor promoter/operator design under tight length constraints; motif-compatibility tradeoff exploration; producing a small, diverse candidate set for assays; workspace-scoped parameter studies with aggregate comparison.
- **Mental model:** deterministic data prep (`fetch`/`lock`) + strict Gibbs annealing optimization (`sample`) + artifact-native analytics (`analyze`).

### Documentation map

Use the demos as the default entrypoint:

1. [Docs map](docs/README.md)
2. [Two-transcription-factor demo](docs/demos/demo_pairwise.md)
3. [Three-transcription-factor demo](docs/demos/demo_multitf.md)
4. [Project workspace (all TFs)](docs/demos/project_all_tfs.md)
5. [Intent and lifecycle](docs/guides/intent_and_lifecycle.md)
6. [Sampling and analysis](docs/guides/sampling_and_analysis.md)
7. [Ingestion guide](docs/guides/ingestion.md)
8. [MEME Suite guide](docs/guides/meme_suite.md)
9. [Studies guide](docs/guides/studies.md)
10. [Portfolio aggregation](docs/guides/portfolio_aggregation.md)
11. [Artifacts reference](docs/reference/artifacts.md)
12. [Config reference](docs/reference/config.md)
13. [CLI reference](docs/reference/cli.md)
14. [Architecture reference](docs/reference/architecture.md)
