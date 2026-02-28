# USR sync troubleshooting

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


Use this sequence when sync commands fail or verification blocks transfer.

## Failure diagnosis sequence

1. Check remote wiring: `uv run usr remotes doctor --remote <name>`.
2. Re-run with explicit verification mode: `--verify hash|size|parquet`.
3. If `verify=auto` prints fallback warnings, resolve missing remote capabilities instead of ignoring warnings.
4. For dataset directory mode, ensure target has `records.parquet` and a discoverable `registry.yaml` ancestor.
5. For file mode, confirm remote path mapping with `--remote-path` or `remote.repo_root` plus local repo root mapping.
6. If doctor reports `Remote flock is unavailable`, install `flock` (util-linux) on remote host.

## Common failure signatures

- `verify=hash requires remote sha256`: remote host lacks hash utility in `PATH`.
- `verify=parquet requires remote parquet row/col stats`: remote host lacks `pyarrow`.
- `Remote flock is unavailable`: remote host cannot provide cross-host dataset lock for sync transfers.
- `Dataset directory path is outside --root and no registry.yaml ancestor was found`: pass correct `--root` or use dataset id.

## Related runbooks

- Baseline loop: [sync-quickstart.md](sync-quickstart.md)
- Setup and remote wiring: [sync-setup.md](sync-setup.md)
- Target modes and mapping: [sync-modes.md](sync-modes.md)
- Machine-readable decisions: [sync-audit-loop.md](sync-audit-loop.md)
- HPC loop: [hpc-agent-sync-flow.md](hpc-agent-sync-flow.md)
- Chained DenseGen/Infer flow: [chained-densegen-infer-sync-demo.md](chained-densegen-infer-sync-demo.md)
- Adversarial drills: [sync-fidelity-drills.md](sync-fidelity-drills.md)
