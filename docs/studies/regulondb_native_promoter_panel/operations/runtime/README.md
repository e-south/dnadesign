## RegulonDB Runtime Context

`command-groups/pipeline.yaml` stores command groups and downstream surface
bindings for the RegulonDB native promoter panel. It is supplemental runtime
context; the OPS-facing contract is `../ops.study.yaml` plus `../contract/`.

Use `command-groups/README.md` before the full pipeline when the task is
operator navigation. The lane sidecars under `command-groups/lanes/` keep
source intake, USR import, Construct, Infer, and LatentDNA command posture
separate while preserving `pipeline.yaml` as the compatibility payload.
