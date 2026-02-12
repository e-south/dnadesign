## OPAL — Optimization with Active Learning

**OPAL** is an [EVOLVEpro-style](https://www.science.org/doi/10.1126/science.adr6006) active-learning engine for biological sequence design.

- **Train** a top-layer regressor on your chosen representation **X** and label **Y**.
- **Predict** every candidate sample with **X** present.
- **Reduce** each prediction Ŷ to a scalar **score** via your configured **objective** → `pred__y_obj_scalar`.
- **Rank** candidates by that score and **select top-k**.
- **Append** runtime events to **ledger sinks** under **`outputs/`** (per-round predictions + run metadata).
- **Persist** artifacts per round (model, selection CSV, round context, objective meta, logs) for auditability.

The pipeline is plugin-driven: swap **data transforms** (X/Y), **models**, **objectives**, and **selection** strategies in `configs/campaign.yaml` without touching core code.

> Using OPAL day-to-day? See the **[CLI Manual](./docs/reference/cli.md)**

---

### Contents

* [Quick start](#quick-start)
* [What gets saved](#what-gets-saved)
* [How OPAL is wired](#how-opal-is-wired)
* [Campaign layout](#campaign-layout)
* [Docs map](#docs-map)
* [Demo campaign](#demo-campaign)


---

### Quick start

OPAL lives inside the `dnadesign` repo. The CLI entrypoint is defined in `pyproject.toml`:

```toml
[project.scripts]
opal = "dnadesign.opal.src.cli:main"
```

```bash
# 1) Initialize a campaign workspace (creates inputs/, outputs/, state.json, and .opal/config marker)
uv run opal init -c path/to/configs/campaign.yaml

# 2) Validate essentials + X + (if present) Y shape/values
uv run opal validate -c path/to/configs/campaign.yaml

# 3) Ingest experimental labels (CSV/Parquet → Y), preview, then write new columns to X dataset
uv run opal ingest-y -c path/to/configs/campaign.yaml --round 0 --csv path/to/my_labels.csv

# 4) Train on labels with observed_round ≤ R, score the pool, select top-k, persist artifacts & events
uv run opal run -c path/to/configs/campaign.yaml --round 0

# 5) Inspect progress
uv run opal status -c path/to/configs/campaign.yaml
uv run opal status -c path/to/configs/campaign.yaml --with-ledger
uv run opal runs list -c path/to/configs/campaign.yaml
uv run opal log -c path/to/configs/campaign.yaml --round latest
uv run opal record-show -c path/to/configs/campaign.yaml --id <some_id>
uv run opal explain -c path/to/configs/campaign.yaml --round 1
uv run opal plot --list                                           # list available plot kinds
uv run opal plot -c path/to/configs/campaign.yaml
uv run opal plot -c path/to/configs/campaign.yaml --run-id <run_id>  # run-aware; resolves round, conflicts error
uv run opal predict -c path/to/configs/campaign.yaml --out path/to/predictions.parquet
uv run opal objective-meta -c path/to/configs/campaign.yaml --round latest
uv run opal verify-outputs -c path/to/configs/campaign.yaml --round latest
uv run opal notebook -c path/to/configs/campaign.yaml             # list notebooks / nudge next step
uv run opal notebook generate -c path/to/configs/campaign.yaml --round latest
uv run opal notebook generate -c path/to/configs/campaign.yaml --name my_analysis --no-validate
uv run opal notebook run -c path/to/configs/campaign.yaml

# (Optional) Start fresh: remove OPAL-derived columns from records.parquet
uv run opal prune-source -c path/to/configs/campaign.yaml --scope campaign
```

If you run commands from outside this repo checkout, use:

```bash
uv run --project /path/to/dnadesign opal --help
```

**Round terminology**

- **observed_round**: the round stamp recorded when labels are ingested (`opal ingest-y`).
- **labels-as-of**: the training cutoff used by `opal run`/`opal explain` (uses labels with `observed_round ≤ R`).

### What gets saved

* **Per-round**
  - `outputs/rounds/round_<k>/`
    - `model/`
      - `model.joblib`
      - `model_meta.json` *(includes training__y_ops when configured)*
      - `feature_importance.csv`
    - `selection/`
      - `selection_top_k.csv`
      - `selection_top_k__run_<run_id>.csv` *(immutable per-run copy)*
    - `labels/`
      - `labels_used.parquet`
    - `metadata/`
      - `round_ctx.json` *(runtime audit & fitted Y-ops)*
      - `objective_meta.json` *(objective mode/params/keys)*
    - `logs/`
      - `round.log.jsonl`

* **Campaign-wide ledger (append-only)**

  * `outputs/ledger/runs.parquet`
    - Plugin configs, counts, objective summaries, artifact hashes, versions.
  * `outputs/ledger/predictions/`
    - Ŷ vector, scalar score, selection rank/flag, and row-level diagnostics (e.g., logic fidelity/effects).
    - `pred__y_hat_model` is in objective-space.
  * `outputs/ledger/labels.parquet`
    - 1 row per label event (observed round, id, y).

Schemas are **append-only**; uniqueness is enforced for: `run_id` (runs), `(run_id,id)` (predictions). Labels are event rows; exact duplicates are de‑duplicated, but distinct sources for the same `(observed_round,id)` are preserved.

**state.json** tracks campaign state per round, including `run_id` and `round_log_jsonl` paths for auditability.

---

## How OPAL is wired

At the top level, OPAL is a CLI (Typer) over an application layer with plugin registries.

### Code layout

```
src/dnadesign/opal/src/
├─ cli/                     # CLI app + command registry
│  ├─ app.py                # Typer entrypoint
│  ├─ registry.py           # auto-discovers and mounts commands
│  └─ commands/             # run, ingest_y, predict, explain, record_show, status, runs, log, validate, init, plot, prune_source
├─ config/                  # YAML loader + plugin param schemas (Pydantic)
├─ registries/              # transforms_x, transforms_y, models, objectives, selection, plots
├─ transforms_x/            # X transforms (import = register)
├─ transforms_y/            # Y ingests (import = register)
├─ models/                  # model wrappers (e.g., RandomForest)
├─ objectives/              # objective fns (Ŷ → scalar score + diagnostics)
├─ selection/               # selection strategies (scores → ranks/selected)
├─ plots/                   # plot plugins
├─ core/                    # RoundCtx, console helpers, core utils/errors
├─ runtime/                 # run_round, ingest, predict, explain, preflight, round_plan
├─ storage/                 # data_access, ledger, artifacts, writebacks, workspace, state, locks
├─ reporting/               # status, summary, record_show
├─ analysis/                # analysis utilities (dashboard modules under analysis/dashboard)
└─ …
```

Dashboard notebooks (e.g., `prom60_eda.py`) now pull their shared logic from
`src/dnadesign/opal/src/analysis/dashboard/` to keep data contracts explicit and reusable for future dashboards.

---

### Campaign layout

`opal init` scaffolds a campaign folder and ensures the label history column exists in `records.parquet`. Edit
`configs/campaign.yaml` to define behavior.

```
<repo>/src/dnadesign/opal/campaigns/<my_campaign>/
├─ configs/
│  ├─ campaign.yaml
│  └─ plots.yaml                  # optional plot config (recommended)
├─ .opal/
│  └─ config                     # auto-discovery marker (path to configs/campaign.yaml)
├─ records.parquet
├─ state.json
├─ inputs/                       # drop experimental label files here
└─ outputs/
   ├─ ledger/
   │  ├─ predictions/            # append-only run_pred parts
   │  ├─ runs.parquet            # run_meta (deduped)
   │  └─ labels.parquet          # label events (ingest-only)
   └─ rounds/
      └─ round_<k>/
         ├─ model/
         │  ├─ model.joblib
         │  └─ model_meta.json
         ├─ selection/
         │  └─ selection_top_k.csv
         ├─ labels/
         │  └─ labels_used.parquet
         ├─ metadata/
         │  ├─ round_ctx.json
         │  └─ objective_meta.json
         └─ logs/
            └─ round.log.jsonl
```

> **Note:** sequences may live in USR datasets under `src/dnadesign/usr/datasets/<dataset>/records.parquet` and are not copied into campaigns.

---

### Docs map

Core docs now live under `src/dnadesign/opal/docs/` with explicit concept/reference separation:

* Docs hub: [`docs/README.md`](./docs/README.md)
* Concepts:
  * [`docs/concepts/architecture.md`](./docs/concepts/architecture.md)
  * [`docs/concepts/roundctx.md`](./docs/concepts/roundctx.md)
* Reference:
  * [`docs/reference/configuration.md`](./docs/reference/configuration.md)
  * [`docs/reference/data-contracts.md`](./docs/reference/data-contracts.md)
  * [`docs/reference/cli.md`](./docs/reference/cli.md)
  * [`docs/reference/plots.md`](./docs/reference/plots.md)
  * [`docs/reference/plugins/models.md`](./docs/reference/plugins/models.md)
  * [`docs/reference/plugins/selection.md`](./docs/reference/plugins/selection.md)
  * [`docs/reference/plugins/transforms-x.md`](./docs/reference/plugins/transforms-x.md)
  * [`docs/reference/plugins/transforms-y.md`](./docs/reference/plugins/transforms-y.md)
* Objective math references:
  * [`docs/objectives/sfxi.md`](./docs/objectives/sfxi.md)
  * [`docs/objectives/spop.md`](./docs/objectives/spop.md)
* Demo:
  * [`docs/guides/demo-sfxi.md`](./docs/guides/demo-sfxi.md)
* Internal notes:
  * [`docs/internal/journal.md`](./docs/internal/journal.md)
  * [`docs/internal/prom60_sfxi_diagnostics_plots.md`](./docs/internal/prom60_sfxi_diagnostics_plots.md)

---

## More documentation

Use the docs hub for task-oriented navigation: [`docs/README.md`](./docs/README.md).

## Demo campaign

See the **[Demo Guide](./docs/guides/demo-sfxi.md)** for a runnable example using `sfxi_v1` (vec8) with a Random Forest model.

---

@e-south
