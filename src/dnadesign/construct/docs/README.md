## construct docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14

### Read order

1. [Top README](../README.md): package intent and boundary reminders.
2. [Docs index](index.md): compact by-type map.
3. [Workspaces guide](../workspaces/README.md): local scaffold and config conventions.
4. [Dev index](dev/README.md): implementation notes and journal.

### Primary workflow

1. Create or choose a workspace.
2. Prepare a config with an input USR dataset, template source, part placements, and output dataset target.
3. Run `construct validate config --config <path>` for schema checks.
4. Run `construct validate config --config <path> --runtime` for template/input/preflight checks.
5. Run `construct run --config <path>` or `construct run --config <path> --dry-run`.
6. Feed the resulting USR dataset into downstream tools such as `infer`.

### Boundary reminder

- construct realizes new DNA sequences from explicit specs.
- Replacement workflows should pin explicit coordinates and, when available, the incumbent interval sequence.
- USR stores both the realized sequence records and the `construct__*` lineage overlay columns.
- Future multi-part placement belongs here, not in `infer`, `densegen`, or `ops`.
