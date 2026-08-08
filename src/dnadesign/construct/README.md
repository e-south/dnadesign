![Construct banner](docs/assets/construct-banner.svg)

Construct realizes declared DNA parts inside explicit sequence contexts. A
workspace maps candidate fields into named template slots and records the
resulting spans and lineage for downstream tools. A separate `compose` command
assembles already chosen linear ssDNA segments into a local artifact bundle;
that specialized topology is one Construct route, not the tool's general data
model.

## Documentation

- [Construct docs](docs/README.md): first runs, workspaces, references, and cross-tool handoffs.
- [Getting started](docs/getting-started.md): shortest validated path for a demo run or blank custom workspace.
- [Workspace guide](workspaces/README.md): scaffold a blank workspace or copy a packaged demo profile.
- [Template/context contract](docs/reference/template-contexts.md): downstream `construct__*` fields and anchor-placement metadata.
- [Linear ssDNA composition](docs/reference/linear-ssdna-composition.md): specialized composition of already chosen single-stranded segments.
- [Repository docs index](../../../docs/README.md): repo-wide index for cross-tool workflows.
