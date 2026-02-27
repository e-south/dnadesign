## Cruncher Glossary

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Core terms](#core-terms)
- [Source and ingest terms](#source-and-ingest-terms)
- [Run and analysis terms](#run-and-analysis-terms)

### Core terms
- **workspace**: A directory containing `configs/`, `inputs/`, `.cruncher/`, and `outputs/`.
- **regulator set**: Ordered list of TF names optimized jointly in one sample run.
- **catalog root**: Local cache root (`catalog.root`) with normalized motifs/sites and discovery artifacts.
- **lockfile**: Frozen mapping from TFs to exact source motif artifacts and hashes.
- **parse cache**: Workspace-local parse validation output under `.cruncher/parse/`.

### Source and ingest terms
- **source**: Named ingest adapter namespace (for example `regulondb`, `demo_local_meme`).
- **discovered source**: Discovery output source ID from `cruncher discover motifs` (for example `demo_merged_meme_oops`).
- **motif matrix source**: `catalog.pwm_source=matrix`; use cached motif matrices.
- **site source**: `catalog.pwm_source=sites`; build matrices from cached site sequences.
- **orientation normalization (bidirectional)**: Sequence identity normalization by lexicographic min of sequence and reverse-complement when bidirectional logic is enabled.

### Run and analysis terms
- **run directory**: Output tree for one regulator-set run under `outputs/`.
- **study**: Workspace-scoped sweep spec (`configs/studies/*.study.yaml`) and deterministic output bundle.
- **portfolio**: Cross-workspace aggregation spec (`configs/*.portfolio.yaml`) and deterministic aggregate output bundle.
- **entrypoint artifacts**: Short list of first files to inspect (`summary.json`, `report.md`, `plot_manifest.json`).
