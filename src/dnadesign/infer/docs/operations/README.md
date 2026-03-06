## Infer Operations

### Local Runtime Checks

- Validate config contract: `uv run infer validate config --config <path>`
- Validate USR dataset path/field: `uv run infer validate usr --dataset <id> --field sequence`
- Run dry-run extract for fast contract checks: `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`

### HPC and Orchestration Context

- Infer workflow contracts in ops:
  - `infer_batch_submit`
  - `infer_batch_with_notify_slack`
- Notify/ops integration depends on shared infer USR producer contract parsing in `src/dnadesign/_contracts/usr_producer.py`.

Use this package-local page with repo-wide operations docs:

- `docs/operations/README.md`
- `docs/bu-scc/README.md`
