![baserender banner](assets/baserender-banner.svg)

`baserender` is a contract-first sequence rendering runtime with strict schemas, explicit adapters, and a stable public API at `dnadesign.baserender`. Use it when you want one tool-agnostic rendering surface across multiple upstream records and no silent fallback behavior. The generic orchestration schema is `BaseRenderJobV3` / `RenderJobV3`; explicit render-contract descriptors such as `sequence_rows_render_v3` and `nucleotide_evidence_map_render_v3` name the concrete visualization use case. Historical job names remain compatibility aliases.

## Documentation

- [baserender docs index](docs/README.md): compact route map for reference, integrations, demos, and examples.
- [Technical reference](docs/reference.md): public API, runtime contracts, and renderer surface.
- [Integration contracts](docs/integrations/README.md): tool-specific adapter contracts and expectations, including YIU visual handoff contracts.
- [Workspace demos](docs/demos/workspaces.md): packaged workspace/job path for validation and rendering.
- [Example jobs](docs/examples): checked-in YAML examples for runnable render inputs.
- [Repository docs index](../../../docs/README.md): repo-wide route map for adjacent tools and workflows.
