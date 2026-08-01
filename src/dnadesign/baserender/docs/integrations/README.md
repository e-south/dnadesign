# Integration Contracts

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-01


This section contains tool-specific contracts between upstream producers and `baserender`.

Use these pages when wiring a specific producer schema to BaseRender adapters and public API helpers.

## Available integrations

- `densegen`: `docs/integrations/densegen.md`
- `cruncher`: `docs/integrations/cruncher.md`
- `yiu`: `docs/integrations/yiu.md`

## Scope boundary

- `README.md` stays tool-agnostic.
- `docs/reference.md` defines core architecture and public API boundaries.
- Tool-specific schema mapping and usage live only in this `docs/integrations/` directory.
- The default sibling-tool policy is file-contract-first: producers write JSON or JSONL contracts plus `RenderJobV4` YAML, and consumers use `dnadesign.baserender` public APIs or CLI only.
