<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

# Minimal LatentDNA workspace

Edit `config.yaml` to replace the placeholder input path and embedding column with your data. The source declares row and subject keys; the view selects the numerical representation. This template does not include an input dataset.

In an activated LatentDNA environment, from this directory:

```bash
# Validate the source columns after updating the configuration.
latentdna validate workspace --workspace . --deep

# Inspect the configured views before materializing one.
latentdna inspect views --workspace .
```

Generated artifacts are written under `outputs/`. Use `latentdna --help` to find analysis commands and `latentdna <group> <command> --help` for their options.
