## OPAL notebook support for agents

- Follow repo-root `AGENTS.md`.
- This directory owns ordinary Python support APIs used by generated OPAL
  notebooks; it does not contain the generated marimo notebooks themselves.
- Preserve the public imports from `notebooks.api` and keep generated-notebook
  behavior covered by the notebook and CLI suites.
- Campaign notebook generation and artifact rules are owned by the nearest
  campaign and OPAL instructions, not by this support-module scope.

### Tests
```bash
uv sync --locked
uv run pytest -q src/dnadesign/opal/tests/notebooks \
  src/dnadesign/opal/tests/cli/test_cli_notebook_generate.py
```
