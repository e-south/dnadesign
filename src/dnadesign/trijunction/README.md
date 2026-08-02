![TriJunction banner](assets/trijunction-banner.svg)

TriJunction plans oligos for assembling exact linear DNA sequences with
three-way junctions. It accepts uppercase `ACGT` sequences and produces
deterministic bundles with sequence checks and vendor-neutral order rows,
without overwriting existing work. The implementation is inspired by
Sidewinder but is independent of the authors' software. The method reference
explains the relationship in detail.

## Documentation

- [TriJunction docs](docs/README.md): choose the shortest route for your task.
- [Getting started](docs/getting-started.md): preflight, plan, build, and verify
  one synthetic example.
- [Request shapes](docs/guides/request-shapes.md): describe one target, a shared
  oligo pool, or several independent pools.
- [Scale and quality review](docs/guides/scale-and-review.md): check larger
  requests before writing files and create optional review images separately.
- [Contracts](docs/reference/contracts.md): request, plan, bundle, and failure
  invariants.
- [Method](docs/reference/method.md): geometry, strand formulas, search scores,
  recovery evidence, and the choices that differ from the papers.
- [Sources](docs/reference/sources.md): Sidewinder papers, attribution, and
  implementation scope.
- [Repository docs](../../../docs/README.md): route to sibling tools and shared
  repository contracts.
