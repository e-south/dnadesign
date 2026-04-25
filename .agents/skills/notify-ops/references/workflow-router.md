# Workflow Router

Use this table to choose the next Notify surface without depending on subtree
`AGENTS.md` loading.

| Need | Open first | Verify next |
| --- | --- | --- |
| Choose the right Notify workflow | `docs/notify/README.md` | `docs/notify/usr-events.md` |
| Set up or refresh a watcher profile | `docs/notify/usr-events.md#setup-flow` | `notify profile doctor --profile <profile.json>` |
| Validate mode exclusivity or resolver inputs | `src/dnadesign/notify/docs/reference/command-contracts.md` | `notify setup resolve-events --json` |
| Start a live watcher | `docs/notify/usr-events.md#run-flow` | watcher cursor and spool state |
| Recover delivery failures | `docs/notify/usr-events.md#recover-flow` | `notify spool drain --profile <profile.json> --fail-fast` |
| Run Notify under BU SCC scheduler control | `docs/bu-scc/batch-notify.md` | `.agents/skills/sge-hpc-ops/SKILL.md` |

Guardrails:
- Notify reads USR `.events.log`, not `outputs/meta/events.jsonl`.
- Reusable live delivery in this repo is file-backed: `--secret-source file`
  plus `--secret-ref file://...`.
- One watcher per live lane or destination dataset is the safe default when
  Notify routing and resume posture matter.
