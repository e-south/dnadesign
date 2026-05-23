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
- Public API and cross-tool contract implementation:
  `src/dnadesign/construct/src/interfaces/`
- Configuration schemas and exception contracts:
  `src/dnadesign/construct/src/contracts/`
- Runtime orchestration: `src/dnadesign/construct/src/orchestration/`
- CLI commands: `src/dnadesign/construct/src/cli/commands/`
- Public package surface: `src/dnadesign/construct/__init__.py`
- Module execution surface: `src/dnadesign/construct/__main__.py`
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
  limited to `__init__.py` and `__main__.py`.
- Add new implementation domains as named packages rather than flat root modules.
- Keep `src/dnadesign/construct/src/` itself free of implementation modules
  other than `__init__.py`; place code in semantic domain packages.
- Do not add root-level `cli.py`, `contracts.py`, or `main.py`; route CLI and
  contract implementation through `src/` packages and export public APIs from
  `dnadesign.construct`.
- Keep tests grouped by behavior (`cli`, `runtime`, `package`, contracts) when
  adding broader coverage.
