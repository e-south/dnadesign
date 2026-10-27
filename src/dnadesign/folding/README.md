![folding banner](assets/folding-banner.svg)

Folding predicts secondary structure for assembled sequence artifacts and
renders ViennaRNA plots for review. It can also publish a create-only advisory
assessment of an exact digest-addressed molecular state. Producers own the
state and sequence; Folding owns backend preflight, isolated execution,
prediction parsing, and assessment evidence. Assessment never changes the
producer's validity decision. The result record is backend-neutral, but the
current runner is not a plugin framework: its only implemented backend is
ViennaRNA. Its request contract supports only `temperature_c`; unknown
parameters fail validation instead of being recorded without effect.
Assessment publication requires a non-root POSIX worker with kernel-enforced
no-fork limits. A Python backend runs in that worker, and a CLI backend runs as
its single permitted process; neither backend may create child processes. The
assessment fails closed where that containment cannot be enforced.

## Documentation

- [Docs index](docs/README.md): command routes, artifact flow, backend policy,
  and boundaries.
- [Composition handoff](../../../docs/architecture/decisions/adr-0002-generic-linear-ssdna-composition.md):
  design rationale for Construct/Folding/BaseRender handoffs.
- [Repository docs index](../../../docs/README.md): cross-tool routing for
  workspace tools, stateless services, and shared contracts.
