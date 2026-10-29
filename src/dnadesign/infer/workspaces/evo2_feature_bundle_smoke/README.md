[Infer guide](../../docs/README.md) / [Workspaces](../README.md)

# Evo2 feature-bundle example

This example compares isolated sequences with sequences surrounded by flanks. Its records are short toy inputs for checking configuration and pooling, not experimental data.

- `inputs/anchor_only_records.jsonl` supplies the isolated sequences.
- `inputs/template_1kb_records.jsonl` supplies resolved flanked sequences and anchor coordinates. Despite the filename, these toy inputs are shorter than 1 kb.
- `config.yaml` declares one job for each input set.

From the repository root in an activated Infer environment:

```bash
infer validate config --config src/dnadesign/infer/workspaces/evo2_feature_bundle_smoke/config.yaml
infer run --config src/dnadesign/infer/workspaces/evo2_feature_bundle_smoke/config.yaml --dry-run
```

For an installed package outside the checkout, copy the packaged example into your current directory first:

```python
from importlib.resources import files
from shutil import copytree

source = files("dnadesign.infer").joinpath("workspaces/evo2_feature_bundle_smoke")
copytree(str(source), "evo2_feature_bundle_example")
```

Then validate `evo2_feature_bundle_example/config.yaml`. These commands do not load a model. Before removing `--dry-run`, configure the runtime, device and precision described in the [feature-bundle guide](../../docs/operations/evo2-sequence-features.md).
