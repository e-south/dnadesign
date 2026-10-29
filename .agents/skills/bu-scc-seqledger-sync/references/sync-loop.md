# Sync Loop

Use this short ladder for BU SCC dataset transfers.

| Step | Command or source | Why |
| --- | --- | --- |
| Configure or inspect remote | `src/dnadesign/seqledger/docs/operations/sync/setup.md` | Establish SSH and remotes posture first. |
| Doctor remote | `uv run seqledger remotes doctor --remote <name>` | Fail fast before transfer. |
| Check reusable auth | `uv run seqledger remotes status --remote <name>` | See whether a control socket is already live. |
| Warm auth when Duo or keyboard-interactive is still needed | `uv run seqledger remotes warm-auth --remote <name>` | Complete auth in a real terminal before transfer. |
| Preview diff | `uv run seqledger --remotes-config <remotes.yaml> diff <dataset> <remote-name>` | Confirm action before mutation. |
| Pull or push | `uv run seqledger --remotes-config <remotes.yaml> pull|push <dataset> <remote-name> -y` | Execute the chosen transfer. |
| Verify | `uv run seqledger validate <dataset> --strict` | Confirm dataset contract after sync. |

Guardrails:
- Pull is the default posture unless the user explicitly asks to push.
- Never delete SCC datasets as part of bootstrap or refresh.
- Only treat directories with `records.parquet` as real datasets.
- For a study handoff, read its private workflow before choosing the SeqLedger root
  and dataset. Validate the declared dataset after a bootstrap or major
  refresh.
