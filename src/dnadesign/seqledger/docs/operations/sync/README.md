<!--
**Type:** route
**Plane:** data-plane
**Owner-boundary:** seqledger
**Entry artifact:** sync intent that still needs a task-specific transfer route
**Exit artifact:** chosen sync runbook for setup, execution, or troubleshooting
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-29
-->

[SeqLedger guide](../../README.md) / [Operations](../README.md)

# Transfer datasets over SSH

Use sync to keep a complete dataset on another machine, including its overlays, sequence views and mutation history. `pull` copies remote state to the local root; `push` copies local state to the remote root. Choose the direction for each transfer rather than running both in sequence.

Start with [SSH and remote setup](setup.md), then follow the [transfer quickstart](quickstart.md).

| Task | Guide |
| --- | --- |
| Map a dataset directory or a single file | [Target modes and verification](modes.md) |
| Read transfer decisions in JSON | [Sync audit output](audit-loop.md) |
| Work between a local machine and HPC | [HPC transfer workflow](hpc-agent-flow.md) |
| Diagnose a failed connection or transfer | [Troubleshooting](troubleshooting.md) |
| Test sync changes locally | [Verification drills](fidelity-drills.md) |

Full dataset transfers default to primary-file hash verification, sidecar checks and content hashes for `_derived` and `_auxiliary` files. They do not preserve cross-host file ownership or permission bits. The [verification reference](modes.md#transfer-consistency) explains history checks, locking and partial-transfer options.
