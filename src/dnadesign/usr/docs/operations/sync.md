## USR Sync over SSH

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This page is the sync router for USR dataset and file transfers over SSH.
Use the split runbooks below to follow progressive disclosure by task.

### Read this first

- [sync-quickstart.md](sync-quickstart.md): minimum reliable loop (`diff` -> `pull`/`push`) and iterative HPC batch pattern.
- [sync-setup.md](sync-setup.md): SSH keys, remote profile setup, and key rotation hygiene.
- [sync-modes.md](sync-modes.md): dataset directory mode and file mode path mapping.
- [sync-troubleshooting.md](sync-troubleshooting.md): failure signatures and deterministic diagnosis sequence.

### Default sync contract

- Dataset sync defaults to `--verify hash` plus strict sidecar and `_derived`/`_auxiliary` content-hash fidelity checks.
- Use `--no-verify-sidecars` only when you intentionally trade fidelity for speed.
- Use `--no-verify-derived-hashes` only when you intentionally keep sidecar inventory checks but skip content-hash parity.

### Quick command loop

```bash
# Preview local-vs-remote diff.
uv run usr diff densegen/my_dataset bu-scc
# Pull remote state into local dataset path.
uv run usr pull densegen/my_dataset bu-scc -y
# Push local state back to remote when needed.
uv run usr push densegen/my_dataset bu-scc -y
```

### Related runbooks

- [sync-audit-loop.md](sync-audit-loop.md)
- [hpc-agent-sync-flow.md](hpc-agent-sync-flow.md)
- [chained-densegen-infer-sync-demo.md](chained-densegen-infer-sync-demo.md)
- [sync-fidelity-drills.md](sync-fidelity-drills.md)
