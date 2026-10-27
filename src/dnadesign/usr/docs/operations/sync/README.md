## USR Sync over SSH

**Type:** route
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** sync intent that still needs a task-specific transfer route
**Exit artifact:** chosen sync runbook for setup, execution, or troubleshooting

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14


Use this page to choose setup, daily sync, or troubleshooting for USR dataset and file transfers over SSH.

### Read this first

- [quickstart.md](quickstart.md): minimum reliable loop (`diff` -> `pull`/`push`) and iterative HPC batch pattern.
- [setup.md](setup.md): SSH keys, remote profile setup, and key rotation hygiene.
- [modes.md](modes.md): dataset directory mode and file mode path mapping.
- [troubleshooting.md](troubleshooting.md): failure signatures and deterministic diagnosis sequence.

### Default sync contract

- Dataset sync defaults to `--verify hash` plus strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks.
- Use `--no-verify-sidecars` only when you intentionally trade fidelity for speed.
- Use `--no-verify-derived-hashes` only when you intentionally keep sidecar inventory checks but skip content-hash parity.

### Quick command loop

```bash
# Preview local-vs-remote diff.
uv run usr diff my_dataset bu-scc
# Pull remote state into local dataset path.
uv run usr pull my_dataset bu-scc -y
# Push local state back to remote when needed.
uv run usr push my_dataset bu-scc -y
```

### Related runbooks

- [audit-loop.md](audit-loop.md)
- [hpc-agent-flow.md](hpc-agent-flow.md)
- [chained-densegen-infer-runbook.md](chained-densegen-infer-runbook.md)
- [fidelity-drills.md](fidelity-drills.md)
