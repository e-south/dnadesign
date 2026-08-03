# Sources and scope

**Type:** reference
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

`junction` is an independently developed planner inspired by the Sidewinder
three-way-junction assembly method. Sidewinder names the published method;
PyWinder names software described by the paper authors. Neither name is used as
this package, command, or schema identity.

## Primary literature

1. Noah Evan Robinson, Weilin Zhang, Rajesh Ghosh, Bryan Gerber, Hanqiao Zhang,
   Charles Sanfiorenzo, Sixiang Wang, Dino Di Carlo, and Kaihang Wang.
   “Construction of complex and diverse DNA sequences using DNA three-way
   junctions.” *Nature* (2026).
   [doi:10.1038/s41586-025-10006-0](https://doi.org/10.1038/s41586-025-10006-0).
2. Noah Evan Robinson, Jean-Sebastien Paul, Weilin Zhang, Hanqiao Zhang,
   Sixiang Wang, Tianhua Zhao, Ezri Abraham, Benjamin Simpson, and Kaihang Wang.
   “One-pot parallel Sidewinder construction from oligo pools.” *bioRxiv*,
   version posted May 2, 2026.
   [doi:10.64898/2026.05.01.722326](https://doi.org/10.64898/2026.05.01.722326).

The second source is a preprint and was not peer reviewed in the inspected
version. Both papers report a Caltech patent application covering described
methods. Citation is attribution, not software permission, patent clearance,
regulatory approval, or freedom-to-operate advice.

The Nature paper introduces the three-way-junction assembly geometry,
barcode/coding oligo roles, ligation, recovery, and experimental validation.
The preprint describes pooled oligos, a three-stage string-design procedure,
construct-specific and universal recovery experiments, and downstream
hierarchical assembly.

## What is implemented

`junction` implements its own strict request contract, bounded deterministic
string search, strand composition, string reconstruction checks,
vendor-neutral order rows, create-only bundles, and offline replay. The strand
orientation and domain formulas map to the paper's barcode and coding oligos.

The implementation is not a copy or compatibility layer for PyWinder. No
output-equivalence claim is made. Where the papers leave recurrence, ranking,
tie, stopping, or failure behavior underspecified, method v1 makes a versioned
local choice documented in [Method v1](method-v1.md).

The papers use different oligo-preparation routes. The Nature workflow
phosphorylates coding oligos before pairing them with barcode oligos. The pooled
preprint phosphorylates and anneals the mixed oligo pool. `junction` records
only the caller's complement-strand end-preparation declaration; it does not
specify either complete preparation route or generate a reaction protocol.

## What is not established

- `thermodynamic_screening` is always `not_run`; string separation is not
  thermodynamic orthogonality.
- Paper-reported fidelity, scale, yield, and parameter results apply to the
  tested experiments, not to arbitrary `junction` output.
- The tool does not implement thermodynamic screening, including the
  NUPACK-guided and NUPACK-inspired approaches described by the sources.
- It also does not implement automatic primer design, buffer equalization,
  Type IIS payload removal, degenerate library compilation, laboratory
  execution, supplier submission, or purchasing.
- A shared `universal` primer declaration does not reproduce the preprint's
  complete universal-recovery and hierarchical-assembly architecture.
- The preprint reports that PCR can favor shorter products and can therefore
  enrich a truncated misassembly that retains both terminal priming regions.
  `junction` does not model that amplification bias or turn the paper's
  observed error counts into a general prediction.
- The repository's software tests do not show that a generated oligo set will
  synthesize, anneal, ligate, amplify, clone, or function experimentally.

Optional BaseRender images are original QA views derived from the neutral
review record. They do not reproduce paper figures or add experimental
evidence.

Before laboratory or commercial use, review the primary papers, their
supplementary materials and licenses, the relevant software licenses,
institutional requirements, synthesis constraints, and applicable legal
questions independently.
