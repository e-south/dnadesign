## evo2_feature_bundle_smoke

Small packaged smoke path for the Evo2 feature-bundle contract.

This workspace keeps the repo-aligned boundaries explicit:

- `anchor_only_records.jsonl` contains direct anchor records, including control-style and candidate-style rows
- `template_1kb_records.jsonl` contains already-resolved template-backed contexts plus `construct__*` anchor metadata
- `config.yaml` runs one anchor-only job and one templated job through `feature_bundle`

Use it as the canonical contributor smoke path:

```bash
uv run infer validate config --config src/dnadesign/infer/workspaces/evo2_feature_bundle_smoke/config.yaml
uv run infer run --config src/dnadesign/infer/workspaces/evo2_feature_bundle_smoke/config.yaml --dry-run
```

To switch models, edit one line in `config.yaml`:

```yaml
model:
  id: evo2_20b
```

The packaged records are intentionally short toy sequences so the workspace stays readable and versionable. The contract being exercised is the anchor/template metadata handoff and selector/pooling bundle semantics, not final biological scale or one specific assay domain.
