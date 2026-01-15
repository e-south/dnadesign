# DenseGen Documentation

This directory is the **authoritative, concise** documentation for DenseGen. It is organized to keep
user guidance, reference material, and developer internals **decoupled** so each can evolve without
forcing edits elsewhere.

## Index

- Start here: `guide/demo.md`
- CLI usage: `cli.md`
- Architecture overview: `architecture.md`
- Developer specification (internal contracts): `dev/developer-spec.md`
- Alignment audit (refactor status): `dev/alignment-audit.md`
- Rolling notes (compact): `notes/README.md`
- User guide:
  - `guide/README.md`
  - `guide/demo.md`
  - `guide/inputs.md`
  - `guide/generation.md`
  - `guide/postprocess.md`
  - `guide/outputs-metadata.md`
  - `guide/workspace.md`
- Reference
  - Config schema: `reference/config.md`
  - Output formats: `reference/outputs.md`
- Roadmap / overhaul plan: `roadmap/OVERHAUL_PLAN.md`

## Conventions

- **No fallbacks:** configuration is strict and unknown keys are errors.
- **Schema version:** `densegen.schema_version` is required (currently `2.1`).
- **Run root:** `densegen.run` is required; configs must live inside `densegen.run.root`.
- **Path resolution:** inputs resolve against the config file directory; outputs/logs/plots must resolve inside `densegen.run.root`.
- **Canonical IDs:** Parquet + USR use the same deterministic ID scheme.
- **Output schema:** `output.schema` defines `bio_type` and `alphabet` for all sinks.
- **Metadata:** namespaced as `densegen__<key>` in outputs.
- **USR optionality:** Parquet-only usage must not require USR modules.

## Doc hierarchy

```
docs/
  README.md
  cli.md
  architecture.md
  guide/
    README.md
    inputs.md
    generation.md
    postprocess.md
    outputs-metadata.md
    workspace.md
  dev/
    developer-spec.md
    alignment-audit.md
  notes/
    README.md
  reference/
    config.md
    outputs.md
  roadmap/
    OVERHAUL_PLAN.md
```
