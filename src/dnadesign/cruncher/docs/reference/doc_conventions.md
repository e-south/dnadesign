## Cruncher Doc Conventions

**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Command conventions](#command-conventions)
- [Path conventions](#path-conventions)
- [Metadata conventions](#metadata-conventions)
- [Generated section markers](#generated-section-markers)

### Command conventions
- Standard command prefix: `uv run cruncher ...`
- Optional shell helper is allowed in tutorials:
  - `cruncher() { uv run cruncher "$@"; }`
- Use config path placeholders where practical:
  - `-c configs/config.yaml`

### Path conventions
- Use workspace-relative placeholders in docs:
  - `<workspace>`
  - `<run_dir>`
  - `<study_id>`
  - `<portfolio_id>`
- Avoid user-machine absolute paths (for example `/Users/...`).

### Metadata conventions
Every markdown page must include near the top:
- `Doc kind`
- `Audience`
- `Updated by`
- `Applies to`
- `Last verified`
- `Primary artifacts`

### Generated section markers
Use marker blocks for generated content:

```md
<!-- docs:map:start -->
...generated content...
<!-- docs:map:end -->
```

```md
<!-- docs:runbook-steps:start -->
...generated content...
<!-- docs:runbook-steps:end -->
```
