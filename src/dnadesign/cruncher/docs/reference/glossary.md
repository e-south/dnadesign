## Cruncher Glossary

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Core terms](#core-terms)
- [Cassette workflow terms](#cassette-workflow-terms)
- [Snapback workflow terms](#snapback-workflow-terms)
- [YIU workflow terms](#yiu-workflow-terms)
- [Source and ingest terms](#source-and-ingest-terms)
- [Run and analysis terms](#run-and-analysis-terms)

### Core terms
- **workspace**: A directory containing `configs/`, `inputs/`, `.cruncher/`, and `outputs/`.
- **workflow family**: A first-class Cruncher lane with its own workspace contract, command surface, and artifact model.
- **regulator set**: Ordered list of TF names optimized jointly in one sample run.
- **catalog root**: Local cache root (`catalog.root`) with normalized motifs/sites and discovery artifacts.
- **lockfile**: Frozen mapping from TFs to exact source motif artifacts and hashes.
- **parse cache**: Workspace-local parse validation output under `.cruncher/parse/`.

### Cassette workflow terms
- **cassette spec**: A strict YAML document at `configs/cassettes/<name>.cassette.yaml` describing hairpin topology, duplex context, nick windows, catalog path, and output behavior.
- **nickase catalog**: A strict local YAML document that defines asymmetric recognition sites plus strand and cut-offset metadata for cassette planning.
- **target strand**: The duplex strand (`primary` or `complement`) that both intended nicks must hit. Legacy specs may still refer to this as the designated strand.
- **pair map**: The explicit list of stem-position couplings linking the 5' arm to the reverse-complement 3' arm.
- **bounded nicked segment**: The interval between the two intended nick boundaries on the target strand. It does not imply downstream removal/excision.

### Snapback workflow terms
- **snapback spec**: A strict YAML document at `configs/snapback/<name>.snapback.yaml` describing one explicit single-nick foldback design under `single_nick_snapback_v2`.
- **snapback solve spec**: A strict YAML document at `configs/snapback/<name>.snapback.solve.yaml` describing one bounded search under `single_nick_snapback_solve_v3`.
- **canonical top strand**: The authored reference sequence and coordinate frame used by the snapback lane.
- **nick boundary**: The resolved zero-based closed boundary where the intended nick lands.
- **retained homology**: The nick-anchored segment that remains paired to the foldback arm after nicking. Some visual docs call this the retained stem.
- **source cap sequence**: The sequence already present between the end of retained homology and the end of the canonical top strand.
- **cap sequence**: The authored cap extension appended after the canonical top strand.
- **effective cap sequence**: `source_cap_sequence + cap_sequence`. In the live snapback contract this must total exactly `3 nt`.
- **foldback arm**: The appended sequence that pairs against retained homology in the post-nick foldback state.
- **QA triptych**: The three published snapback states: `pre_nick_duplex`, `post_nick_exposed`, and `post_nick_foldback`.

### YIU workflow terms
- **YIU spec**: A strict YAML document at `configs/yiu/<name>.yiu.yaml` describing the source oligo, ordered step graph, retained payload goal, cleanup assumptions, and output settings.
- **state graph**: The ordered list of intended molecular states emitted by the explicit YIU tracer bullet, from `source_oligo_ssdna` through `downstream_amplifiable_product`.
- **retained product**: The intended sequence or connected region set that survives nicking, cleanup, foldback, and adapter ligation.
- **sacrificial region**: A source-oligo interval intended to be fragmented and depleted during nickase digestion and cleanup.

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
