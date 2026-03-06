## infer operations

### Start path

1. Validate command contracts locally:
   - `uv run infer validate config --config <path>`
   - `uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run`
2. Run pressure-test path from this index.
3. Verify dataset state with `uv run usr ...` before enabling scheduler submit.

### Runbooks

- [Agnostic model + USR pressure test](pressure-test-agnostic-models.md): standalone and ops-runbook paths.
- [End-to-end pressure-test demo](../tutorials/demo_pressure_test_usr_ops_notify.md): infer + usr + ops + notify full walkthrough.

### Integration boundary

- Ops workflow contracts consumed by infer:
  - `infer_batch_submit`
  - `infer_batch_with_notify_slack`
- Shared producer contract path:
  - `src/dnadesign/_contracts/usr_producer.py`

### Related docs

- [infer docs index](../README.md)
- [repository operations index](../../../../../docs/operations/README.md)
