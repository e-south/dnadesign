![Construct banner](docs/assets/construct-banner.svg)

Construct takes anchor sequences or other focal DNA parts and realizes them
inside explicit larger sequence contexts. It can assemble one or more named
slots from a candidate row into one template, preserving slot-level spans for
downstream tools. It also composes declared linear ssDNA segment specs into
local artifact bundles when the user already has the parts and does not need a
workspace-backed solve. Any sequence record can serve as an anchor, a template,
or a helper part; `construct` makes that role explicit in config and records the
resulting lineage for downstream tools.

## Documentation

- [Construct docs](docs/README.md): first runs, workspaces, references, and cross-tool handoffs.
- [Getting started](docs/getting-started.md): shortest validated path for a demo run or blank custom workspace.
- [Linear ssDNA composition](docs/reference/linear-ssdna-composition.md): validate and run declared segment composition specs without creating new workspaces.
- [Workspace guide](workspaces/README.md): scaffold a blank workspace or copy a packaged demo profile.
- [Template/context contract](docs/reference/template-contexts.md): downstream `construct__*` fields and anchor-placement metadata.
- [Repository docs index](../../../docs/README.md): repo-wide index for cross-tool workflows.
