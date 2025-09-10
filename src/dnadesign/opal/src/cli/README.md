## OPAL CLI Overview

How things are wired (high level):
```bash
pyproject.toml
  opal = "dnadesign.opal.src.cli.app:main"
                    │
                    ▼
src/cli/app.py (Typer app)
  - calls discover_commands() and install_registered_commands()
  - runs Typer app()

src/cli/registry.py
  - @cli_command(...) decorator used by each command module
  - discover_commands(): import cli/commands/*
  - install_registered_commands(app): mount onto Typer

src/cli/commands/*.py
  - Thin wrappers around application modules
  - Share helpers from commands/_common.py
    (config discovery, store construction, error reporting, json_out)

src/*.py (application layer)
  - explain.py, ingest.py, predict.py, preflight.py, status.py, writebacks.py, etc.
  - No Typer here; return Python objects or raise OpalError

src/registries.py (plugin registry)
  - TRANSFORMS (ingest CSV→Y)
  - OBJECTIVES (Ŷ→scalar score)
  - SELECTIONS (scores→ranks/selected)

src/transforms/*, src/objectives/*, src/selection/*
  - Modules register themselves via decorators at import time
```

```bash
src/cli/
  app.py            # builds Typer app, root callback, Ctrl-C handling
  registry.py       # @cli_command decorator, discovery, install into app
  commands/
    _common.py      # shared helpers: resolve_config_path, store_from_cfg, json_out, internal_error
    init.py
    ingest_y.py
    run.py
    explain.py
    predict.py
    model_show.py
    record_show.py
    status.py
    validate.py
```

- Each command file exports one function decorated with @cli_command("name", help="…").
- Command modules do not implement business logic; they call into src/*.py application modules.
