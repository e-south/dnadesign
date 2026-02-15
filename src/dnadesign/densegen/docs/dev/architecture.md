## DenseGen architecture (short)

DenseGen is a staged pipeline with explicit artifacts between stages. Each stage consumes
typed config and emits outputs that can be inspected or replayed.

---

### Pipeline

```
config -> Stage-A pools -> Stage-B libraries -> dense-arrays solver
       -> postprocess -> outputs + plots + notebooks
```

---

### Modules (high level)

- `config/` - schema + validation (no silent fallbacks).
- `adapters/sources/` - Stage-A input adapters.
- `core/sampler.py` + `core/artifacts/*` - Stage-A pools + Stage-B libraries.
- `adapters/optimizer/` - solver wrappers.
- `core/pipeline.py` - orchestration + runtime guards.
- `core/metadata*` + `adapters/outputs/*` - metadata + sinks.
- `viz/*` - plots and notebook-facing visual artifacts.

---

### Contracts

- Config resolution is explicit and fails fast on ambiguity.
- Outputs stay under `densegen.run.root/outputs`.
- Stage-A pools are cached; Stage-B is the only resampling stage.
- Metadata is validated against a schema; no silent drift.

---

### Extension points

- New inputs: add a source adapter and wire it into the source factory.
- New plots: register in `viz/plotting.py`.
- New postprocess steps: implement under `core/postprocess/` and record policy metadata.

---

@e-south
