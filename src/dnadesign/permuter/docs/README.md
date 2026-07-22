# Permuter Docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-24

Use Permuter when a reference sequence needs explicit nucleotide, codon, or
multi-site variants with attached model or heuristic scores.

## Start Here

- [Architecture](architecture.md): ownership boundary, identity semantics, and
  fail-fast rules for complementary use with USR, Construct, Infer, studies,
  and Ops.
- [CLI and data contracts](cli-and-data-contracts.md): command surface, workspace
  scopes, dataset layout, evaluators, and output columns.
- [Handoffs](handoffs.md): USR materialization, non-executing Infer feature
  requests, and study-owned candidate promotion boundaries.
- [Modernization plan](modernization-plan.md): living dev spec for contract,
  workspace, and public API hardening.
- [RT variant generation](rt_variant_generation.md): multi-mutation RT variant
  construction from single-amino-acid DMS results.
- [RT variant selection](rt_variant_selection.md): score-gated,
  diversity-constrained selection of multi-site variants.
