## OPS failure contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-27

This page defines the maintainer-facing CLI failure contract for `ops`.

### Exit codes

| Exit code | Meaning |
| --- | --- |
| `0` | success |
| `2` | usage, validation, contract, or path-policy failure |
| `1` | unexpected internal failure |

OPS does not use a large custom exit-code taxonomy.

### Output rules

For failures:

- plain-text error text belongs on stderr
- messages should be plain text
- the message should not depend on TTY-only formatting
- stdout should stay empty unless the command intentionally emits a machine-readable failure payload
- the installed `dnadesign.ops.cli:main` wrapper should duplicate and close stderr per invocation, not keep a process-lifetime extra fd open after import

### Error categories

Contract and usage failures should be concise and actionable:

- unknown command or invalid enum
- unknown registry id
- missing required input
- malformed manifest or invalid path contract
- runbook plan or runbook execute contract failure

Unexpected failures should still fail fast, but should be surfaced as internal errors rather than silently swallowed.

Scheduler probes are part of the contract surface too:

- live `qstat`-backed probes must fail fast rather than hanging indefinitely
- timeout-driven scheduler failures should surface as explicit contract errors or degraded probe records
- fixture mode should stay deterministic and must not silently degrade when the fixture path is invalid
- supported scheduler diagnostics live under `uv run ops runbook diagnostics ...`
- `ops runbook plan`, `ops runbook active-jobs`, and `ops runbook execute` must expose explicit runtime-visibility state instead of silently converting unknown scheduler posture into `no active jobs`
- `ops runbook execute --submit` must fail closed by default when active-job visibility is unknown, unless the user passes `--allow-unknown-active-jobs`
- internal gate helpers may still exist for implementation, but public operator docs and skill packs should not require `python -m dnadesign.ops.orchestrator.gates ...`

### Examples

Usage failure:

```text
Usage: ops [OPTIONS] COMMAND [ARGS]...
Try 'ops --help' for help.

Error: No such command 'bogus'.
```

Contract failure:

```text
Error: Catalog contract error: unknown registry id: missing.registry
```

### Maintainer checks

When changing OPS CLI parsing, entrypoint wiring, or error rendering, run:

```bash
uv run pytest -q src/dnadesign/ops/tests/test_cli_failure_contract.py
uv run pytest -q src/dnadesign/ops/tests/test_sge_gates.py -k run_native_gate_command_surfaces_failure_text_to_stderr
uv run pytest -q src/dnadesign/ops/tests/test_runbook_orchestrator.py -k captures_gate_stderr_for_nonzero_native_gate_command
uv run pytest -q src/dnadesign/ops/tests/test_runbook_orchestrator.py -k "active_job_visibility_is_unknown or named_preset_for_project_defaults"
```

Keep `CliRunner` checks and subprocess-facing checks aligned so operator-facing and machine-captured failures report the same truth.
