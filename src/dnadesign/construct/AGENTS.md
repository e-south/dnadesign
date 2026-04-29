## `construct` for agents

Supplement to the repo-root `AGENTS.md` for Construct template realization,
anchor handling, focal windows, and sequence-view product contracts.

## Boundaries

- Treat Construct as a generic sequence construction tool. Promoter anchors,
  core windows, and pDual templates are study inputs, not package-level
  semantics.
- Express products through explicit contracts such as anchor-only inserts,
  realized template contexts, orientations, pooling intervals, and derived
  analysis windows.
- Keep centering, truncation, and expansion rules data-driven through annotated
  features and workspace config. Fail fast when required focal annotations are
  absent or ambiguous.
- Preserve source metadata from input records unless a documented transform
  intentionally derives or removes a field.

## Key paths

- Tool README: `src/dnadesign/construct/README.md`
- Runtime source: `src/dnadesign/construct/src/`
- CLI commands: `src/dnadesign/construct/src/cli/commands/`
- Contracts and public wrappers: `src/dnadesign/construct/contracts.py`,
  `src/dnadesign/construct/cli.py`
- Workspaces: `src/dnadesign/construct/workspaces/`
- Tests: `src/dnadesign/construct/tests/`

## Generated artifacts

- Treat workspace `outputs/`, `runs/`, and batch result directories as generated.
  Fix config or code and rerun Construct instead of hand-editing products.
- Ask before committing generated datasets or large binary artifacts.

## Commands

```bash
uv run construct --help
uv run construct validate --help
uv run construct run --help
uv run pytest -q src/dnadesign/construct/tests
```

## Layout

- Keep runtime implementation under `src/`; root-level Python files should stay
  limited to stable import and CLI wrappers.
- Add new implementation domains as named packages rather than flat root modules.
- Keep tests grouped by behavior (`cli`, `runtime`, `package`, contracts) when
  adding broader coverage.
