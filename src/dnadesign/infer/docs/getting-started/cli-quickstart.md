<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Infer guide](../README.md)

# Validate your first Infer job

Create a workspace, inspect its sequence input and check an extraction job. These commands do not download weights or produce predictions, so you can run them without a GPU.

## 1. Install and activate Infer

From a DNADesign checkout, install the configuration and dry-run profile:

```bash
uv sync --locked --no-default-groups --extra infer
source .venv/bin/activate
```

If Infer is already installed, activate that environment instead. The commands below use its `infer` executable and can run outside the checkout. See [installation](../../../../../docs/setup/installation.md) for other profiles.

## 2. Create a workspace

Run this in a directory where you want to keep your work:

```bash
infer workspace init --id first_infer --profile local
cd workspaces/first_infer
cat inputs/records.jsonl
```

The starter input contains one record:

```json
{"id":"example_record","sequence":"ACGTACGT"}
```

Open `config.yaml` to inspect the job. It reads `inputs/records.jsonl`, requests Evo2 logits and averages them over nucleotide positions. The input path is relative to the configuration file, so moving the whole workspace preserves that relationship.

## 3. Validate before inference

```bash
# Check the model, job settings and requested output.
infer validate config --config config.yaml

# Also resolve the input and display the planned job.
infer run --config config.yaml --dry-run
```

The commands finish with `Config validated.` and `Config validated (dry run).`, respectively. A dry run checks the request and input availability; it does not establish that the model can execute on this machine.

## 4. Try an ad-hoc request

Use a preset to validate extraction directly from a DNA string:

```bash
infer extract --preset evo2/extract_logits_ll --seq ACGT --dry-run
```

For a generation request, specify a prompt and output length:

```bash
infer generate \
  --model-id evo2_7b \
  --device cpu \
  --precision bf16 \
  --alphabet dna \
  --prompt ACGT \
  --max-new-tokens 4 \
  --dry-run
```

These finish with `Extract validated (dry run).` and `Generate validated (dry run).`. Use `infer presets list` and `infer adapters list` to inspect available presets and registered models.

## Continue with your data

- [Workspaces](../../workspaces/README.md) explains file formats and dataset paths.
- [Evo2 feature bundles](../operations/evo2-sequence-features.md) explains context, pooling and the runtime needed for actual inference.
- [SeqLedger extraction](../operations/pressure-test-agnostic-models.md) adds persistent outputs and resumable processing.

For a configuration error, compare the reported field with the [command reference](../reference/command-contracts.md). Changing the model or input does not require starting a new workspace.
