## infer CLI quickstart

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-25

These checks validate configuration and output requests without loading a model
or producing predictions. Run them from the repository root after
[installation](../../../../../docs/setup/installation.md).

### 1) Validate a config contract

```bash
uv run infer validate config --config src/dnadesign/infer/docs/operations/examples/pressure_test_infer_config.yaml
```

### 2) Run extract in dry-run mode

```bash
uv run infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run
```

### 3) Run generate in dry-run mode

```bash
uv run infer generate \
  --model-id evo2_7b \
  --device cpu \
  --precision bf16 \
  --alphabet dna \
  --prompt ACGT \
  --max-new-tokens 4 \
  --dry-run
```

Success prints `Config validated.`, `Extract validated (dry run).`, and
`Generate validated (dry run).`. These checks do not download model weights or
verify that inference can run on this machine.

### 4) Next route

- Move to [operations pressure-test runbook](../operations/pressure-test-agnostic-models.md) for end-to-end usage.
