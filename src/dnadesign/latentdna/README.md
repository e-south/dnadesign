![latentdna banner](docs/assets/latentdna-banner.svg)

LatentDNA is the comparison layer for `dnadesign`.

LatentDNA compares learned sequence representations and turns those comparisons
into workspace-owned tables, plots, snapshots, and notebooks. Study-specific
labels live in workspace config, while downstream benchmarking and final
selection stay with the owning study or consumer tool.

## Documentation

- [LatentDNA docs](docs/README.md): workflow routing, contracts, concepts, and operations.
- [Workspaces guide](workspaces/README.md): packaged templates and local workspace scaffolding.
- [CLI contracts](docs/reference/cli-contracts.md): current command surface and machine output contracts.
- [Workspace schema](docs/reference/workspace-schema.md): `latentdna.workspace.v1` core config shape.
- [Workspace snapshot contract](docs/reference/workspace-snapshot-contract.md): study-facing status surface.
- [Artifact naming grammar](docs/reference/artifact-naming.md): canonical representation, scope, and deliverable naming.
- [Promoter-study representation comparison workflow](docs/workflows/promoter-study-representation-comparison.md): active promoter-study comparison path.
- [Context-geometry workflow](docs/workflows/context-shift.md): paired anchor-versus-full-context geometry metrics.
- [Cross-view agreement workflow](docs/workflows/cross-view-agreement.md): structural agreement diagnostics outside the promoter-study path.
- [Repository docs index](../../../docs/README.md): repo-wide upstream and downstream handoff index.
