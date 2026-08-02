# TriJunction Docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

Use TriJunction when you have exact linear uppercase `ACGT` target sequences
and need a deterministic, inspectable plan for three-way-junction oligos. It
does not create or own a workspace. Callers own requests and durable
destinations; TriJunction owns validation, string-level design, publication,
and offline verification.

## Start Here

- [Getting started](getting-started.md): run the shortest complete CLI path.
- [Request shapes](guides/request-shapes.md): decide whether targets belong in
  one shared physical pool or independent pool design streams.
- [Scale and quality review](guides/scale-and-review.md): map realistic request
  sizes to fail-closed preflight and optional downstream visual review.
- [Contract reference](reference/contracts.md): inspect accepted fields,
  invariants, schemas, and bundle contents.
- [Method reference](reference/method.md): inspect locus geometry, strand
  orientation, string objectives, recovery evidence, and explicit v1 choices.
- [Sources and scope](reference/sources.md): trace the Sidewinder lineage and
  understand what this implementation does not claim.
- [Repository docs index](../../../../docs/README.md): choose a sibling tool or
  cross-tool route.

## Choose an Operation

| Need | Command | Durable write |
| --- | --- | --- |
| Validate and exercise the full design search | `uv run trijunction preflight <request>` | no |
| Inspect the complete plan | `uv run trijunction plan <request>` | no |
| Publish one create-only bundle | `uv run trijunction build <request> --output <new-directory>` | yes |
| Recompute and verify a bundle | `uv run trijunction verify <bundle>` | no |

TriJunction performs deterministic string-level planning. It does not run
thermodynamic screening, choose experimental conditions, submit orders, or
establish that a design will work in the laboratory. Direct synthesis belongs
to a separate workflow rather than a fallback inside this lifecycle. Optional
BaseRender quality-assurance views are downstream projections; they do not add
a second TriJunction command path or mutate a verified bundle.
