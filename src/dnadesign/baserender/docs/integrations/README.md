---
owner: dnadesign-maintainers
last_verified: 2026-08-10
---

# Integrations

Choose the tool that wrote the input. Each guide names the accepted record,
adapter, renderer, and failure rules.

## Available integrations

- [DenseGen](densegen.md)
- [Cruncher](cruncher.md)
- [YIU](yiu.md)
- [junction](junction.md)

## Scope boundary

- Integrations translate producer records; they do not move producer analysis
  into BaseRender.
- Built-in integrations live under `src/integrations/<producer>/` and register
  through one internal descriptor catalog.
- The central job parser reads descriptors and contains no producer branches.
- External plugin discovery is intentionally absent. Add an entry-point pack
  only when an integration is distributed independently.
- Use the public `dnadesign.baserender` API or CLI. Private `src.*` imports are
  unsupported.

Run `uv run baserender catalog --json` to inspect the installed capability set.
