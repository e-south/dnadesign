# TriJunction Docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

Use TriJunction to turn exact linear uppercase `ACGT` sequences into checked
plans for three-way-junction oligos. You provide the request and choose where
to save the result. TriJunction validates the request, designs the sequences,
writes a new bundle, and can verify that bundle offline. It does not manage a
workspace or copy external study records. Target and pool IDs are saved in the
bundle, so use identifiers that are safe for its intended destination.

## Start Here

- [Getting started](getting-started.md): run the shortest complete CLI path.
- [Request shapes](guides/request-shapes.md): decide which targets will share
  an oligo pool and which should be designed independently.
- [Scale and quality review](guides/scale-and-review.md): test larger requests
  against the current limits and create optional review images.
- [Contract reference](reference/contracts.md): inspect accepted fields,
  invariants, schemas, and bundle contents.
- [Method reference](reference/method.md): inspect locus geometry, strand
  orientation, search scores, recovery evidence, and v1 design choices.
- [Sources and scope](reference/sources.md): trace the Sidewinder lineage and
  understand what this implementation does not claim.
- [Repository docs index](../../../../docs/README.md): choose a sibling tool or
  cross-tool route.

## Choose an Operation

| Need | Command | Writes files? |
| --- | --- | --- |
| Validate and exercise the full design search | `uv run trijunction preflight <request>` | no |
| Inspect the complete plan | `uv run trijunction plan <request>` | no |
| Publish one bundle in a new directory | `uv run trijunction build <request> --output <new-directory>` | yes |
| Recompute and verify a bundle | `uv run trijunction verify <bundle>` | no |

TriJunction checks sequences and reconstructs each target. It does not run
thermodynamic screening, choose experimental conditions, submit orders, or
show that a design will work in the laboratory. It also does not switch short
targets to direct synthesis. BaseRender can turn the saved review records into
optional quality-assurance images without changing the verified design bundle.
