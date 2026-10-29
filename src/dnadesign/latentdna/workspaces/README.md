<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA](../README.md) / [Guide](../docs/README.md)

# Organize a workspace

A workspace keeps one analysis configuration beside its inputs and outputs. Input files may live elsewhere; generated artifacts always live under the workspace's `outputs/` directory.

```text
my-analysis/
├── config.yaml
├── inputs/              # Optional local inputs
└── outputs/             # Generated matrices, plots, manifests and run records
```

For a complete runnable example, follow [Run a small local analysis](../docs/getting-started/README.md).

## Start from a template

The installed package includes a `minimal` template. In an activated LatentDNA environment:

```bash
# Create a new workspace from the packaged starter.
latentdna workspace init --template minimal --workspace ./my-analysis
```

Edit `my-analysis/config.yaml` to point at your data and vector column before validating. The template's input path is a placeholder, not a bundled dataset.

```bash
# Check the completed configuration against its source schema.
latentdna validate workspace --workspace ./my-analysis --deep
```

Caller-supplied relative workspace paths resolve from the current working directory. Source paths such as `inputs/sequences.parquet` resolve from the workspace. Use an absolute workspace path when invoking commands from several directories.

You can also copy your own template with `--template /path/to/template`. See the [workspace schema](../docs/reference/workspace-schema.md) for fields and [external study workspaces](../../../../docs/integrations/study-workspaces.md) for optional study bindings.
