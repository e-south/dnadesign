## Cruncher to DenseGen PWM workflow (artifact-first)

This workflow demonstrates a decoupled path where Cruncher exports PWM artifacts
and DenseGen consumes them via `pwm_artifact_set`. The DenseGen config remains the
single source of truth for runtime sampling behavior.

### 1) Ensure Cruncher has motifs cached

Use the demo workspace if you want a reproducible example:

```bash
uv run cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR -c src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml
```

### 2) Export DenseGen motif artifacts (one file per motif)

```bash
uv run cruncher catalog export-densegen \
  -c src/dnadesign/cruncher/workspaces/demo_basics_two_tf/config.yaml \
  --tf lexA --tf cpxR --source demo_local_meme \
  --out /path/to/densegen-run/inputs/motif_artifacts \
  --background record \
  --pseudocount 0.01
```

This writes per-motif JSON files plus an `artifact_manifest.json` for inspection. Any directory
works; keeping artifacts under the DenseGen run `inputs/` directory keeps the workspace
self-contained.

### 3) Point DenseGen at the artifacts

```yaml
densegen:
inputs:
    - name: lexA_cpxR
      type: pwm_artifact_set
      paths:
        - inputs/motif_artifacts/demo_local_meme__lexA.json
        - inputs/motif_artifacts/demo_local_meme__cpxR.json
      sampling:
        strategy: stochastic
        n_sites: 80
        oversample_factor: 10
        score_percentile: 90
        length_policy: exact
```

PWM sampling is stochastic. Under schema `2.2+`, `pool_strategy: subsample` will resample
reactively on stalls/duplicate guards, while `iterative_subsample` resamples proactively
after `arrays_generated_before_resample` or when a library under-produces.

### 4) Run DenseGen

```bash
pixi run dense validate-config -c path/to/config.yaml
uv run dense inspect config -c path/to/config.yaml
uv run dense run -c path/to/config.yaml --no-plot
```

### Captured output (excerpt)

```
INFO | dnadesign.densegen.src.core.pipeline | PWM input sampling for lexA_cpxR: motifs=2 | sites=lexA x 80, cpxR x 80 | strategy=stochastic | score=percentile=90 | oversample=10 | length=exact
INFO | dnadesign.densegen.src.core.pipeline | Library for lexA_cpxR/lexA_cpxR: 16 motifs | TF counts: lexA x 8, cpxR x 8 | target=180 achieved=192 pool=subsample
INFO | dnadesign.densegen.src.core.pipeline | [lexA_cpxR/lexA_cpxR] 8/8 (100.00%) (local 8/8) CR=1.050 | seq ...
```

DenseGen writes `outputs/meta/inputs_manifest.json` plus run-scoped library artifacts under `outputs/`
(`outputs/attempts.parquet`),
capturing resolved PWM sampling settings and the exact TFBS library offered to the solver.

---

@e-south
