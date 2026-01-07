## Workspaces

Put cruncher workspaces in this folder. Each workspace should contain a
`config.yaml` plus any inputs you want to keep alongside it.

Example (demo workspace):

```
src/dnadesign/cruncher/workspaces/
  demo/
    config.yaml
    .cruncher/        # local cache + lockfiles (generated)
    runs/             # run outputs (parse/sample/analyze/report)
```

Tip: `cd` into a workspace and run cruncher commands without passing `--config`.
