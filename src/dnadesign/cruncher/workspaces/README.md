## Workspaces

Put cruncher workspaces in this folder. Each workspace should contain a
`config.yaml` plus any inputs you want to keep alongside it.

Example:

```
src/dnadesign/cruncher/workspaces/
  demo_basics_two_tf/
    config.yaml
    inputs/                   # demo inputs
    .cruncher/                # local cache + lockfiles (generated)
    outputs/                  # run outputs (parse/sample/analyze/report)
```

Tip: `cd` into a workspace and run cruncher commands without passing `--config`. You can also run from anywhere with `--workspace <name>` or inspect what is available via `cruncher workspaces list`.
