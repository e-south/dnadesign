## Snapback Phenomenology Dev Spec

**Status:** implemented BspQI-pinned retained-active screen
**Audience:** Cruncher maintainers and Snapback study operators
**Study:** `retron_hairpin_design`
**Primary lane:** released-product Snapback
**Contrast lane:** YIU remains contrast-only, not a topology engine

### Problem

The original `de033` default screen reported no exact origin-0, stem-3, cap-3
hit, even though mechanism-valid solutions could be written by hand for
`Nt.BsmAI`, `Nt.BstNBI`, `Nt.AlwI`, `Nb.BtsI`, and `Nb.BsrDI`. The current
study route has closed that gap by pinning the Type IIS release enzyme to
`BspQI` and evaluating retained active top and bottom products.

This is not only a search-route problem. The local catalog must also preserve
vendor cut-site semantics: for `Nb.BtsI` and `Nb.BsrDI`, NEB reports
`GCAGTG(none/0)` and `GCAATG(none/0)`, where the bottom-strand nick is at the
end of the listed six-base recognition motif, not at the left edge of the
expanded `...NN` footprint. The catalog now records that raw notation with an
explicit `motif_end` offset basis. The relevant motif and vendor-diagram records
are:

- `Nt.BsmAI`: `GTCTC`, vendor diagram `GTCTCNN`, source
  <https://www.neb.com/en-us/products/r0121-ntbsmai>
- `Nt.BstNBI`: `GAGTC`, vendor diagram `GAGTCNNNNN`, source
  <https://www.neb.com/en-us/products/r0607-ntbstnbi>
- `Nt.AlwI`: `GGATC`, vendor diagram `GGATCNNNNN`, source
  <https://www.neb.com/en-us/products/r0627-ntalwi>
- `Nb.BtsI`: `GCAGTG`, vendor diagram `GCAGTGNN`, source
  <https://www.neb.com/en-us/products/r0707-nbbtsi>
- `Nb.BsrDI`: `GCAATG`, vendor diagram `GCAATGNN`, source
  <https://www.neb.com/en-us/products/r0648-nbbsrdi>

The failure is semantic. The default search encodes one route family:
`bottom_active_from_top_nick`. It searches exposed-bottom products from
top-strand nicks only, with pre-cut footprint constraints handled conservatively.
The hand-validated mechanisms require a more general processing ontology:
oriented nickase footprints can be reverse-complemented into the design frame,
the retained active product may be top or bottom, degenerate motif positions can
be design variables, and a downstream restriction enzyme can trim a longer
single-stranded product to the final foldback boundary.

The operational command that exercises the fixed ontology is:

```bash
uv run cruncher snapback released-target-search \
  --workspace-root src/dnadesign/cruncher/workspaces/de033 \
  --nick-preset neb_nicking_v1 \
  --nick-additional-preset thermo_nicking_v1 \
  --release-preset type_iis_release_v1 \
  --release-variant-id BspQI \
  --nick-boundary 0 --paired-bp 3 --cap-nt 3 \
  --allow-top-active-routes \
  --allow-precut-footprint-outside-active-product \
  --max-results 16 --near-boundary-search-limit 8
```

Observed result: `exact_hits_found`, with BspQI-pinned exact hits for
`Nt.BstNBI`, `Nt.AlwI`, `Nt.BsmAI`, `Nb.BsrDI`, and `Nb.BtsI`. Exact hits reject
non-degenerate nickase footprint upstream of logical origin 0 because that would
hide retained protected sequence from the realized stem burden. The maintained
lesson from the original defect is that these route semantics must remain
first-class in the design objective.

### Design Objective

Cruncher must answer this question directly:

> Given a nickase footprint, an optional downstream release enzyme, and a target
> snapback topology, can enzymatic processing produce a released single-stranded
> product whose logical origin is 0, stem length is 3, and cap length is 3?

The answer must be independent of whether the retained product is physical top
or bottom, whether the vendor diagram is used forward or reverse-complemented,
and whether the bases satisfying the stem complement come from fixed motif
bases, degenerate motif bases, user-defined payload bases, or downstream
extension trimmed by the release enzyme.

### Required Ontology

#### Coordinate Frames

Add explicit coordinate frames instead of overloading one boundary value:

- `vendor_site_frame`: vendor 5'->3' recognition-site diagram and cut offsets.
- `precursor_top_frame`: actual precursor top-strand coordinates.
- `physical_strand_frame`: top/bottom strand identity after orientation.
- `retained_product_frame`: the strand that remains after the nick and release.
- `logical_snapback_frame`: origin-0, stem/cap/foldback coordinates used for
  target evaluation.
- `foldback_frame`: paired representation after intramolecular foldback.

Every result should expose frame transforms, rather than only final sequences.

#### Processing Events

Model enzymatic processing as a sequence of typed events:

- `RecognitionFootprint`: oriented motif, vendor diagram, fixed-vs-degenerate
  positions, and source catalog entry.
- `NickEvent`: physical nicked strand, boundary in precursor frame, and retained
  strand after nick.
- `FragmentationEvent`: retained fragment and sacrificial fragment after nick.
- `ReleaseCutEvent`: top and bottom release cuts plus the active product
  terminal boundary.
- `ProductProjection`: maps precursor bases into retained-product coordinates.
- `FoldbackProjection`: maps retained-product bases into stem, cap, and return
  arm.

#### Constraint Provenance

Each base in the retained product and foldback topology must carry provenance:

- `fixed_motif_base`
- `degenerate_motif_base`
- `release_motif_base`
- `user_payload_base`
- `synthesized_extension_base`
- `trimmed_by_release`
- `sacrificial_after_nick`

This is the key to explaining why `Nt.BstNBI`, `Nt.AlwI`, and `Nt.BsmAI` can
work: their degenerate tracks can satisfy the stem or foldback-complement
constraints. It also explains the `Nb.BtsI` and `Nb.BsrDI` mode, where fixed
nickase footprint bases can form the first stem and the release enzyme supplies
the distal terminal boundary.

#### Mechanism Classes

Every hit should be classified into one of these mechanism classes:

- `degenerate_footprint_snapback`: degenerate nickase positions are assigned to
  satisfy stem/foldback complementarity.
- `fixed_footprint_plus_release_trim`: fixed nickase footprint contributes
  target stem/cap bases, and a downstream release enzyme trims the product to
  the desired foldback boundary.
- `mixed_footprint_payload`: fixed and degenerate footprint bases plus payload
  extension jointly satisfy the topology.
- `comparison_visual_only`: explicit non-screen example such as
  `msd-HOPV5_snapback`; not a catalog hit.

### Target Semantics

Replace ambiguous fields like `nick_boundary_from_left` as the sole target with
a richer target contract:

```yaml
target_topology:
  logical_origin: 0
  stem_bp: 3
  cap_nt: 3
  retained_product_strands: [top, bottom]
  allow_oriented_vendor_footprints: true
  allow_degenerate_motif_assignment: true
  allow_release_trim_after_foldback_return: true
  require_wc_stem_pairing: true
  require_complete_downstream_fragment_separation: true
```

`logical_origin: 0` means the released product's foldback-relevant sequence
starts at product coordinate 0. It does not mean the physical nickase
recognition site must have no protected or degenerate bases left of the cut in
the vendor diagram. Those are separate frame-transform facts.

### Search Algorithm

1. Enumerate nickase footprints in both orientations.
2. Derive physical nicked strand and retained strand after orientation.
3. Project the retained fragment into `retained_product_frame`.
4. Enumerate release placements as terminal-boundary candidates, not merely as
   downstream cuts attached to one active strand policy.
5. Build a constraint-satisfaction problem over product bases:
   - fixed motif symbols constrain allowed bases;
   - `N` and other IUPAC symbols remain assignable domains;
   - foldback pairing adds Watson-Crick constraints;
   - release cut fixes the product end.
6. Solve the CSP for origin-0, stem-3, cap-3.
7. Emit exact hits when the logical topology is satisfied, regardless of active
   physical strand.
8. Emit near hits only when the logical topology itself drifts, not when the
   current code's route family excludes a valid retained strand.

### Current CLI Contract

Use the released-product target-search route for the read-only probe:

```bash
uv run cruncher snapback released-target-search \
  --workspace-root src/dnadesign/cruncher/workspaces/de033 \
  --nick-preset neb_nicking_v1 \
  --nick-additional-preset thermo_nicking_v1 \
  --release-preset type_iis_release_v1 \
  --release-variant-id BspQI \
  --nick-boundary 0 \
  --paired-bp 3 \
  --cap-nt 3 \
  --allow-top-active-routes \
  --allow-precut-footprint-outside-active-product \
  --json
```

Do not add a separate `screen` alias for this study unless it preserves the
same retained-active route semantics through the public released-product
contract.

### Output Contract

Each exact hit must report:

- enzyme pair and route family
- physical nicked strand
- retained product strand
- oriented vendor footprint
- logical origin, stem span, cap span, foldback return span
- release terminal boundary
- mechanism class
- per-base provenance ledger
- frame transforms from vendor site to precursor to retained product
- rendered precursor, released-product, and foldback plots

The table must separate:

- `logical_stem_bp`: requested and satisfied stem length
- `upstream_retained_duplex_bp`: retained duplex left of the logical origin
- `effective_foldback_pairing_bp`: visual/physical pairing after including
  retained duplex

This prevents a boundary-3 near hit from being mistaken for a true origin-0
stem-3 design, and prevents a true retained-active origin-0 hit from being
hidden as a non-default route.

### Acceptance Tests

Golden exact-hit tests must assert that the following enzymes are found for
origin-0, stem-3, cap-3 in the real catalog screen:

- `Nt.BsmAI`
- `Nt.BstNBI`
- `Nt.AlwI`
- `Nb.BtsI`
- `Nb.BsrDI`

For each, assert:

- `hit_kind == "exact"`
- `logical_origin == 0`
- `logical_stem_bp == 3`
- `cap_nt == 3`
- `mechanism_class` is populated
- at least one product base has correct provenance
- foldback pairing is Watson-Crick valid
- release cut separates downstream sacrificial sequence

Negative tests:

- fixed motif conflicts reject with `FOOTPRINT_CONSTRAINT_UNSAT`
- release cuts inside required topology reject with
  `RELEASE_OVERLAPS_REQUIRED_TARGET_REGION`
- missing frame transform rejects with a fail-fast schema error
- YIU route cannot be selected as a topology solver

### Migration Plan

1. Introduce ontology models without changing ranking:
   `CoordinateFrameTransform`, `ProcessingEventLedger`,
   `ProductBaseProvenance`, and `SnapbackMechanismClass`.
2. Refactor current route flags into explicit target semantics:
   `allow_retained_strands`, `use_vendor_footprints`, and
   `allow_degenerate_motif_assignment`.
3. Add the new screen command as a tracer-bullet path over the existing
   evaluator.
4. Promote retained-active exact hits into the default study screen once golden
   tests pass.
5. Update visuals to draw the mechanism ledger, including which positions are
   fixed motif bases, degenerate assignments, payload bases, and release-trimmed
   termini.

### Non-Goals

- Do not turn YIU into the topology solver.
- Do not rank bench practicality, yield, buffer compatibility, or ligation
  performance in this ontology pass.
- Do not silently treat near hits as exact hits.
- Do not hide valid retained-active mechanisms behind default-off flags once the
  screen command exists.

### Readiness Definition

The implementation is ready when a fresh `de033` released-product probe, using
real nickase and release presets, reports exact origin-0, stem-3, cap-3 hits for
the hand-validated enzyme family and emits a mechanism ledger explaining why
each hit works.

The current implementation slice is the released-product target-search probe.
It preserves the released-product evaluator while keeping retained top and
bottom active-product routes, oriented vendor footprints, and the
origin-0/stem-3/cap-3 target topology explicit in the command contract.
