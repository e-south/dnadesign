![baserender banner](assets/baserender-banner.svg)

`baserender` renders schema-backed sequence visuals from explicit job files or
the stable `dnadesign.baserender` API. It is the linear/component evidence
renderer for producer-owned contracts; it does not solve biological designs or
own secondary-structure layout.

Jobs use `BaseRenderJobV3` / `RenderJobV3`. Render-contract descriptors such
as `sequence_rows_render_v3` and `nucleotide_evidence_map_render_v3` name the
concrete visualization use case. Historical job names remain compatibility
aliases.

## Documentation

- [baserender docs index](docs/README.md): compact route map for reference, integrations, demos, and examples.
- [Technical reference](docs/reference.md): public API, runtime contracts, and renderer surface.
- [Integration contracts](docs/integrations/README.md): tool-specific adapter contracts and expectations, including YIU visual handoff contracts.
- [Workspace demos](docs/demos/workspaces.md): packaged workspace/job path for validation and rendering.
- [Example jobs](docs/examples): checked-in YAML examples for runnable render inputs.
- [Repository docs index](../../../docs/README.md): repo-wide route map for adjacent tools and workflows.
