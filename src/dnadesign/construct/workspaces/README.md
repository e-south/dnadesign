## construct workspaces

Use this directory as the default root for construct workspaces.

### Quick start

```bash
uv run construct workspace init --id demo_construct
```

This creates:

- `workspaces/<id>/config.yaml`
- `workspaces/<id>/inputs/anchor_manifest.template.yaml`
- `workspaces/<id>/inputs/README.md`
- `workspaces/<id>/inputs/template.fa`
- `workspaces/<id>/outputs/logs/ops/audit/`
- `workspaces/<id>/outputs/usr_datasets/`

### Contract

- Workspace ids must be directory names, not paths.
- Existing workspaces are never overwritten.
- Relative config paths resolve from the directory that contains `config.yaml`.
- The output USR root can be bootstrap-created on first construct run when no registry exists yet.
- The scaffold intentionally does not check in canonical long plasmid sequences from chat paste; replace placeholders with file-backed canonical FASTA inputs before real runs.
