## `notify` for agents

Supplement to repo-root `AGENTS.md` with `notify`-specific operator defaults.

### Key paths
- Code: `src/dnadesign/notify/`
- Tests: `src/dnadesign/notify/tests/`
- Canonical operators runbook: `docs/notify/usr-events.md`
- Module-local index: `src/dnadesign/notify/README.md`

### Default low-friction flow

Run setup once per tool/workspace, then run watch by tool/workspace shorthand.

```bash
# discover workspace names for shorthand mode
uv run notify setup list-workspaces --tool <tool>

# one-time webhook setup (config/profile independent)
uv run notify setup webhook --secret-source auto --name <shared-name> --json

# one-time profile setup (stores/reuses webhook and writes profile)
uv run notify setup slack \
  --tool <tool> \
  --workspace <workspace-name> \
  --secret-source auto \
  --secret-ref <webhook-ref>

# validate
uv run notify profile doctor --profile src/dnadesign/<tool-family>/workspaces/<workspace-name>/outputs/notify/<tool>/profile.json

# day-to-day watch (no --profile path needed)
uv run notify usr-events watch --tool <tool> --workspace <workspace-name> --follow
```

Explicit fallback when shorthand is not applicable:
- `notify ... --config <workspace-config.yaml>`

Auto-profile mode resolves profile path as:
- `<config-dir>/outputs/notify/<tool>/profile.json`

If that file does not exist, `notify usr-events watch --tool ... --workspace ...` fails with an actionable setup command.
If profile `events_source` exists but disagrees with the current `--tool/--workspace` (or `--tool/--config`), watch fails fast; rerun setup with `--force`.

### Secret handling
- Preferred: `--secret-source auto` (uses keychain/secretservice when available, otherwise secure local file secrets).
- Env fallback: `--secret-source env --url-env <ENV_VAR>`.
- Never store plaintext webhook URLs in profile files.

### Contracts
- Notify input contract is Universal Sequence Record `.events.log` JSONL only.
- DenseGen runtime diagnostics (`outputs/meta/events.jsonl`) are out of scope.
- `profile_version` must be `2`.
