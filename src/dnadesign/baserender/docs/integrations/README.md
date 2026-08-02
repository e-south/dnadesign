# Integration Contracts

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02


These guides connect outputs from other dnadesign tools to BaseRender.

Choose the tool that wrote the records you want to render.

## Available integrations

- [DenseGen](densegen.md)
- [Cruncher](cruncher.md)
- [YIU](yiu.md)
- [junction](junction.md)

## Scope boundary

- The package README remains general.
- The [technical reference](../reference.md) defines the public API.
- Each integration page explains its tool's fields and gives an example.
- Other tools write JSON or JSONL records and `RenderJobV4` YAML. Render them
  through the public `dnadesign.baserender` API or CLI.
