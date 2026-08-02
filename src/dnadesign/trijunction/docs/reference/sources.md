# Sources and Implementation Scope

**Type:** reference
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

TriJunction is this repository's name for an independently developed planner
inspired by the Sidewinder three-way-junction assembly method and its
pooled-oligo extension. It is not the name of the published method or the
authors' software.

## Primary Literature

1. Noah Evan Robinson, Weilin Zhang, Rajesh Ghosh, Bryan Gerber, Hanqiao Zhang,
   Charles Sanfiorenzo, Sixiang Wang, Dino Di Carlo, and Kaihang Wang.
   “Construction of complex and diverse DNA sequences using DNA three-way
   junctions.” *Nature* (2026).
   [doi:10.1038/s41586-025-10006-0](https://doi.org/10.1038/s41586-025-10006-0).
2. Noah Evan Robinson, Jean-Sebastien Paul, Weilin Zhang, Hanqiao Zhang,
   Sixiang Wang, Tianhua Zhao, Ezri Abraham, Benjamin Simpson, and Kaihang Wang.
   “One-pot parallel Sidewinder construction from oligo pools.” *bioRxiv*
   preprint (2026).
   [doi:10.64898/2026.05.01.722326](https://doi.org/10.64898/2026.05.01.722326).

The first paper introduces Sidewinder assembly through DNA three-way
junctions. The second describes pooled, parallel construction and a
string-based barcode-design approach. Those papers provide the scientific
basis and motivate TriJunction's sequence search. The pooled paper demonstrates
both construct-specific (its term) and universal PCR recovery. TriJunction
records those choices as `target_specific` and `universal` primer declarations;
it does not turn them into laboratory protocols.

## What TriJunction Adds

TriJunction is not Sidewinder, PyWinder, or an official implementation from the
paper authors. This repository defines its schemas, deterministic search
budgets, search records, new-directory publication, and offline verification.

The implementation reports thermodynamic screening as `not_run`. It does not
reproduce the papers' experimental validation, guarantee assembly
fidelity, prescribe a laboratory protocol, select a supplier, or submit an
order. The review records are bound by digest to the verified TriJunction
bundle. Optional BaseRender images use an original visual composition rather
than reproducing paper figures and remain in a separate new bundle. A generated
bundle or image is a review input, not evidence that an assembly is safe,
feasible, licensed, or experimentally validated.

Citation provides attribution, not license, patent, regulatory, or freedom-to-
operate clearance. Review the publications, applicable software licenses,
institutional requirements, and legal constraints independently before use.
