# Sync Loop

Use this short ladder for BU SCC dataset transfers.

| Step | Command or source | Why |
| --- | --- | --- |
| Configure or inspect remote | `src/dnadesign/usr/docs/operations/sync/setup.md` | Establish SSH and remotes posture first. |
| Doctor remote | `uv run usr remotes doctor --remote <name>` | Fail fast before transfer. |
| Check reusable auth | `uv run usr remotes status --remote <name>` | See whether a control socket is already live. |
| Warm auth when Duo or keyboard-interactive is still needed | `uv run usr remotes warm-auth --remote <name>` | Complete auth in a real terminal before transfer. |
| Preview diff | `uv run usr --remotes-config <remotes.yaml> diff <dataset> <remote-name>` | Confirm action before mutation. |
| Pull or push | `uv run usr --remotes-config <remotes.yaml> pull|push <dataset> <remote-name> -y` | Execute the chosen transfer. |
| Verify | `uv run usr validate <dataset> --strict` | Confirm dataset contract after sync. |

Guardrails:
- Pull is the default posture unless the user explicitly asks to push.
- Never delete SCC datasets as part of bootstrap or refresh.
- Only treat directories with `records.parquet` as real datasets.
- For RT-lnRNA Infer handoff work, sync the owning study USR root
  `workspaces/studies/rt_lnrna_sponging_construct_triage/usr` and validate
  `rt_lnrna_sponging_construct_triage_construct_contexts_2000bp_v1` after a
  bootstrap or major refresh.
