![baserender banner](assets/baserender-banner.svg)

`baserender` turns typed sequence and declared-topology records into
deterministic images. It owns reusable nucleotide layout, annotation, styling,
and image publication. The tool that produced a record still owns its metrics,
rankings, interpretation, and analysis plots.

## Documentation

- [Documentation](docs/README.md): choose a job, integration, or demo route.
- [Technical reference](docs/reference.md): public contracts, execution flow,
  and extension boundaries.
- [Integrations](docs/integrations/README.md): producer-specific input mappings.
- [Workspace demos](docs/demos/workspaces.md): validate and run packaged jobs.
- [Example jobs](docs/examples): small `RenderJobV4` YAML examples.

## Start

```bash
uv run baserender catalog
uv run baserender job validate path/to/job.yaml
uv run baserender job run path/to/job.yaml
```

`catalog --json` lists the same adapters, transforms, style profiles,
renderers, and render contracts for scripts and agent tooling. Validation
resolves every declared input before rendering. A successful run publishes one
create-only bundle with a manifest.
