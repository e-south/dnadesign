## Stress Runtime Context

`command-groups/pipeline.yaml` stores command groups and downstream surface
bindings for the stress/ethanol/cipro study. It is supplemental runtime
context; the OPS-facing contract is `../ops.study.yaml` plus `../contract/`.

Use `command-groups/README.md` before opening the full pipeline. The lane
sidecars under `command-groups/lanes/` are operator navigation overlays; the
canonical machine-readable payload stays in `pipeline.yaml` as the shared input
for status, preflight, LatentDNA, OPAL, and USR docs-contract checks.
