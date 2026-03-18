## promoter_evo2_smoke

Small packaged smoke path for the Evo2 promoter feature bundle.

This workspace keeps the repo-aligned boundaries explicit:

- `anchor_only_promoters.jsonl` contains direct promoter anchors, including wildtype and designed rows
- `template_1kb_promoters.jsonl` contains already-resolved template-backed contexts plus `construct__*` anchor metadata
- `config.yaml` runs one anchor-only job and one templated job through `feature_bundle`

Use it as the canonical contributor smoke path:

```bash
uv run infer validate config --config src/dnadesign/infer/workspaces/promoter_evo2_smoke/config.yaml
uv run infer run --config src/dnadesign/infer/workspaces/promoter_evo2_smoke/config.yaml --dry-run
```

To switch models, edit one line in `config.yaml`:

```yaml
model:
  id: evo2_20b
```

The packaged records are intentionally short toy sequences so the workspace stays readable and versionable. The contract being exercised is the anchor/template metadata handoff, not final biological scale.
