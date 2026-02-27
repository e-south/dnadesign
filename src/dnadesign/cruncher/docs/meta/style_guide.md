## Cruncher Docs Style Guide

**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Purpose](#purpose)
- [Required doc structure](#required-doc-structure)
- [Progressive disclosure pattern](#progressive-disclosure-pattern)
- [Runbook coupling rules](#runbook-coupling-rules)
- [Generated sections](#generated-sections)

### Purpose
Use this guide to keep Cruncher docs legible, deterministic, and runnable for cruncher-maintainers.

### Required doc structure
- Add a single top-of-page status line: `> **Last updated by:** cruncher-maintainers on YYYY-MM-DD`.
- Include a `Contents` heading unless the page explicitly opts out with `<!-- docs:toc:off -->`.
- Use relative links inside Cruncher docs; avoid absolute paths and raw GitHub blob links.

### Progressive disclosure pattern
- Keep one “happy path” command near the top.
- Follow with “customize/deep dive” sections.
- Add explicit “Stop here if you only need the happy path” notes for long guides.
- Route repeated details to standard references instead of repeating long lists.

### Runbook coupling rules
- Demos and workspace guides should point to `configs/runbook.yaml`.
- Step-by-step command blocks should preserve machine runbook step order.
- Use stable step IDs when documenting partial execution with `--step`.

### Generated sections
- `docs/README.md` and `docs/index.md` include generated map blocks.
- `docs/reference/runbook_steps.md` is generated from workspace machine runbooks.
- Refresh generated docs with:
  - `uv run python -m dnadesign.cruncher.devtools.docs_ia`
