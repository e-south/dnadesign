# Workload Reference: dnadesign (examples, not core assumptions)

This file provides **examples** for dnadesign-style workloads (CPU solver runs, GPU inference, watchers),
but the core skill remains tool-agnostic.

BU SCC policy and canonical templates are in sibling platform docs:
- `../../README.md`
- `../../jobs/README.md`

## Discover dnadesign BU SCC docs and templates safely

Never assume paths like `docs/bu-scc/...` exist at the repo root. Discover.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Common candidate roots
for d in \
  "$REPO_ROOT/docs/bu-scc" \
  "$REPO_ROOT/dnadesign/docs/bu-scc" \
  "$REPO_ROOT"/*/docs/bu-scc

do
  [ -d "$d" ] && echo "[found] $d"
done

# Discover templates
find "$REPO_ROOT" -maxdepth 8 -type f -path "*/docs/bu-scc/jobs/*" -print | sed -n '1,200p'
```

## Map dnadesign workloads to portable patterns

- CPU solver or pipeline run: CPU multi-thread batch
- GPU inference: GPU batch (or interactive if permitted)
- Notify or event watcher: watcher or daemon batch job
- Model or dataset prefetch: transfer or download-only job (site-specific)

## dnadesign preflight examples (optional)

These are examples of the pattern "validate config before long runs".

```bash
# Example: DenseGen config validation
uv run dense validate-config --probe-solver -c <config.yaml>
uv run dense inspect config --probe-solver -c <config.yaml>
```

## Event watching example (principle)

The principle: the watcher should follow a **single append-only event log** and deliver notifications.
If using dnadesign + USR + Notify, the log is typically a `.events.log` JSONL file.

Keep watchers lightweight, restart-safe, and durable:

- persistent cursor or offset
- optional spool directory for failed deliveries
- minimal CPU and memory

## Template submission example (discover first)

Once templates are located (for example `densegen-cpu.qsub`), submit using absolute paths:

```bash
TEMPLATE="/abs/path/to/.../docs/bu-scc/jobs/<template>.qsub"
qsub -terse <ACCOUNT_ARG> "$TEMPLATE"
```

If the template needs environment variables, pass them explicitly at submit time:

```bash
qsub -terse <ACCOUNT_ARG> \
  -v CONFIG=/abs/path/to/config.yaml \
  "$TEMPLATE"
```

## Guardrails

- Keep dnadesign examples here. Do not reintroduce dnadesign-only assumptions into core skill logic.
- Prefer discovery and variables over hard-coded repo paths.
