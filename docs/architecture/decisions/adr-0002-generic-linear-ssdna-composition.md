## ADR 0002: Generic linear ssDNA composition in Construct

**Status:** accepted
**Date:** 2026-05-13
**Owner:** dnadesign-maintainers

### Context
The retron hairpin design effort needs to compose whole multicopy linear ssDNA
products from smaller sequence pieces. The upstream primitives are already
split by owner: Cruncher Snapback handles cap/shortening geometry, Cruncher
`scar_nick` handles Type IIS scar plus terminal nick feasibility, and the
Retron study record owns biological rationale and selected variants.

Construct already owns deterministic sequence construction, but its current
runtime centers on template-anchor and normalization workflows. This
composition problem needs ordered segment lineage, repeated units, annotations,
and local handoff artifacts rather than anchor/window lineage.

### Decision
Construct owns a new generic `linear_ssdna_composition_v1` workflow.

The ADR records the architecture decision. The Construct reference doc is the
current generic operator authority for commands, bundle ownership, and
Construct/Folding/BaseRender handoff details.

The workflow:

- parses a strict shared contract under `dnadesign.contracts.sequence`
- concatenates ordered physical segments into `linear_ssdna` products
- preserves physical `segment_spans` separately from semantic
  `annotation_spans`
- expands repeated units deterministically with copy-level spans
- validates declared reverse-complement transforms and assertions
- writes local artifact bundles before any optional USR persistence
- emits FASTA, GenBank, feature CSV, and `sequence_evidence_map_v1` sidecars

Folding remains a separate backend-neutral contract under
`dnadesign.contracts.folding` and a separate runtime package under
`dnadesign.folding`. Construct emits a typed folding request and may invoke the
public folding API, but BaseRender remains a consumer of visual contracts and
does not run folding or assemble sequences.

### Rationale
- keeps Construct generic rather than Retron-specific
- preserves Cruncher as the primitive solver boundary
- makes sequence, span, and provenance artifacts inspectable without requiring
  USR schema decisions in the first slice
- keeps scar-nick source refs as provenance/projection inputs rather than raw
  sequence imports
- creates a typed handoff to later folding and rendering phases
- uses uv to lock the official ViennaRNA Python interface for reproducible
  folding while keeping system-provided ViennaRNA `RNAfold` CLI availability
  explicit

### Consequences
- What becomes easier
  - manual and study-authored multicopy ssDNA composition
  - Benchling handoff through GenBank plus sidecars
  - BaseRender-ready component-span visualization
  - explicit preflight, version capture, and advisory missing-backend states
    for the ViennaRNA Python API and the ViennaRNA `RNAfold` CLI interface
  - later USR import from stable local artifacts

- What becomes harder
  - Construct has a second runtime family beside template-anchor jobs
  - documentation and CLI examples need to distinguish `construct run` from
    `construct compose run`
  - operators should use the uv-managed `viennarna` package for default local
    folding; system-provided ViennaRNA `RNAfold` executables remain an optional
    operational dependency for CLI-specific requests

### Boundaries
- Construct must not import Cruncher private modules.
- Cruncher artifacts are consumed later through declared source refs, public
  contracts, files, or USR rows.
- YIU remains contrast-only for the retron hairpin study.
- Folding is advisory by default until a future config makes it a hard gate.
- Folding request execution must go through `dnadesign.folding` public API or
  `uv run folding`, not BaseRender or Construct internals.

### Links
- Proposal:
  `docs/dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md`
- Generic operator authority:
  `src/dnadesign/construct/docs/reference/linear-ssdna-composition.md`
- Implementation record:
  `docs/exec-plans/completed/2026-05-13-generic-linear-ssdna-composition.md`
- Study handoff:
  `docs/studies/retron_hairpin_design/contexts/linear-ssdna-composition.md`
- Implementation:
  `src/dnadesign/construct/src/composition.py`
  `src/dnadesign/folding/`
