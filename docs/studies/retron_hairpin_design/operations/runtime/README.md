## Retron Runtime Context

`command-groups/pipeline.yaml` stores command groups and native-agent bootstrap
metadata for the Retron study. It is supplemental runtime context; the
OPS-facing contract is `../ops.study.yaml` plus `../contract/`.

Use `command-groups/README.md` before the full pipeline when the task is
operator navigation. The lane sidecars under `command-groups/lanes/` keep
compiler, materialize, Snapback, scar-nick, and YIU command posture separate
while preserving `pipeline.yaml` as the compatibility payload.
