## Scar-Nick Workflow

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-08

### Contents
- [Purpose](#purpose)
- [Command Route](#command-route)
- [Workspace Shape](#workspace-shape)
- [Outputs](#outputs)
- [Boundaries](#boundaries)
- [Related Docs](#related-docs)

### Purpose

Use the scar-nick workflow when the task is to evaluate Type IIS retained-scar
base junctions with an exact terminal nick. It is a Cruncher design family for
bounded base-junction feasibility, not a top-level dnadesign tool and not a
phenotype predictor.

### Command Route

```bash
uv run cruncher scar-nick validate --spec <workspace>/configs/scar_nick/<name>.scar_nick.yaml
uv run cruncher scar-nick design --spec <workspace>/configs/scar_nick/<name>.scar_nick.yaml
uv run cruncher scar-nick show --run <workspace>/outputs/scar_nick/<name>
```

Use `validate` for schema, release-enzyme, nickase, and pair-profile checks.
Use `design` only after validation passes; it writes the deterministic run
bundle and BaseRender job handoffs. Use `show` to inspect one explicit run and
fail fast on missing or drifted artifacts.

### Workspace Shape

Scar-nick specs live under one workspace:

```text
<workspace>/
  configs/scar_nick/<panel>.scar_nick.yaml
  outputs/scar_nick/<panel>/
```

Use multiple specs inside one workspace for related panels. Do not create one
workspace per hit or one workspace per left/right base pair.

### Outputs

A successful `design` run writes:

- `meta/scar_nick_manifest.json`
- `meta/scar_nick_status.json`
- `provenance/spec.snapshot.yaml`
- `analysis/reports/report.md`
- `export/table__scar_nick_candidates.csv`
- `export/table__scar_nick_candidate_pair_calls.csv`
- `export/table__scar_nick_nickase_geometry_audit.csv`
- `analysis/views/*.scar_nick_visual.v1.json`
- `baserender_jobs/scar_nick_terminal_nick.job.yaml`

Generated outputs are review artifacts. Update the spec or source code and
regenerate; do not hand-edit the output bundle.

### Boundaries

- Scar-nick owns retained-scar and terminal-nick feasibility.
- Snapback owns single-nick foldback cap and shortening geometry.
- YIU remains a payload-centric contrast route.
- External workflows may consume scar-nick artifacts through the public bundle
  contract. Their interpretation and selection policy stay outside Cruncher.

### Related Docs

- [scar_nick package map](../../src/scar_nick/README.md)
- [Nickase catalog reference](../reference/nickase_catalog.md)
- [Cruncher CLI reference](../reference/cli.md)
