# Sources and Implementation Scope

**Type:** reference
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

TriJunction is an independent, study-neutral implementation inspired by the
Sidewinder three-way-junction assembly method and its pooled-oligo extension.
The name TriJunction describes this repository's planning surface; it is not
the name of the method reported in the papers.

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
string-based barcode-design approach. TriJunction uses these publications as
scientific and algorithmic lineage for its explicit string-level planning
contract. The pooled paper demonstrates both construct-specific (its term) and
universal PCR recovery; TriJunction represents those as neutral
`target_specific` and `universal` recovery-primer declarations rather than
experimental protocols.

## Independent Implementation Boundary

TriJunction is not the authors' Sidewinder or PyWinder software and is not
presented as an official implementation. Its schemas, deterministic search
budgets, evidence receipts, create-only publication, and offline verification
are repository-owned contracts.

The implementation deliberately reports thermodynamic screening as `not_run`.
It does not reproduce the papers' experimental validation, guarantee assembly
fidelity, prescribe a laboratory protocol, select a supplier, or submit an
order. The neutral review records are a digest-bound part of the verified
TriJunction bundle. Optional BaseRender-rendered images use an original visual
composition rather than reproducing paper figures and remain in a separate
create-only render bundle. A generated bundle or image is a review input, not
evidence that an assembly is safe, feasible, licensed, or experimentally
validated.

Citation provides attribution, not license, patent, regulatory, or freedom-to-
operate clearance. Review the publications, applicable software licenses,
institutional requirements, and legal constraints independently before use.
