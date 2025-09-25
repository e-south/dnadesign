# OPAL — Optimization with Active Learning

**OPAL** is an [EVOLVEpro-style](https://www.science.org/doi/10.1126/science.adr6006) active-learning engine for DNA/protein sequence design. It fits a top-layer regressor (e.g., `RandomForestRegressor`) on a chosen representation column **X** and a label column **Y**, predicts **Ŷ** across a candidate set, and **selects the top-k** per round. Campaigns swap plugins (transforms, models, objectives, selection) without touching core code.

---

## Contents

* [Quick install](#quick-install)
* [How OPAL is wired](#how-opal-is-wired)
  * [CLI overview](#cli-overview)
* [Code layout](#code-layout)

* [Campaign layout](#campaign-layout)
* [Configuration (`campaign.yaml`)](#configuration-campaignyaml)

  * [Key blocks](#key-blocks)
  * [Minimal example](#minimal-example)

* [Architecture & data flow](#architecture--data-flow)

  * [RoundContext (per-round state)](#roundcontext-perround-state)
* [Data contracts](#data-contracts)

  * [Records schema (USR essentials)](#records-schema-usr-essentials)
  * [Per-round write-backs](#per-round-write-backs)

* [Demo campaign](#demo-campaign)

---

## Quick install

OPAL lives inside the `dnadesign` repo. The CLI entrypoint is defined in `pyproject.toml`:

```toml
[project.scripts]
opal = "dnadesign.opal.src.cli.app:main"
```

Install in editable mode (from the repo root):

```bash
uv pip install -e .
# or: pip install -e .
```

---

## How OPAL is wired

At the top level, OPAL is a CLI front-end (Typer) over an application layer that uses plugin registries. Commands stay thin; all logic lives in modules that return Python objects.

```text
pyproject.toml
  opal = "dnadesign.opal.src.cli.app:main"
                    │
                    ▼
src/dnadesign/opal/src/cli/app.py                # Typer app entrypoint
  ├─ discover_commands() / install_registered_commands()
  └─ runs the CLI

src/dnadesign/opal/src/cli/commands/*.py         # thin CLI wrappers; share helpers in _common.py
src/dnadesign/opal/src/*.py                      # application layer (ingest, run, predict, status, artifacts, …)
src/dnadesign/opal/src/registries/               # plugin registries (transforms_x/y, models, objectives, selections)
src/dnadesign/opal/src/{transforms_x,transforms_y,models,objectives,selection}/*
                                                 # plugins register via decorators at import time
```

### CLI overview

For full command reference, flags, and common workflows, see the [**CLI Guide**](./src/cli/README.md).

---

## Code layout

```text
src/dnadesign/opal/src/
├─ cli/                     # CLI app + command registry
│  ├─ app.py                # Typer entrypoint
│  ├─ registry.py           # auto-discovers and mounts commands
│  └─ commands/             # plug-in commands (run, ingest_y, predict, explain, record_show, status, validate, init)
├─ config/                  # YAML loader + plugin param schemas (Pydantic)
├─ registries/              # registries for transforms_x, transforms_y, models, objectives, selections
├─ transforms_x/            # concrete X transforms (import = register)
├─ transforms_y/            # concrete Y ingests (import = register)
├─ models/                  # model wrappers (e.g., RandomForest)
├─ objectives/              # objective functions (Ŷ → scalar score + diagnostics)
├─ selection/               # selection strategies (scores → ranks/selected)
├─ artifacts.py             # round artifacts (metrics, logs, ctx, CSVs)
├─ data_access.py           # RecordsStore: IO, label history, fixed-width X
├─ round_context.py         # RoundContext + fingerprinting helpers
├─ writebacks.py            # minimal per-row columns writer
└─ …
```

---

## Campaign layout

`opal init` scaffolds a campaign folder. You then edit `campaign.yaml` to define behavior.

```text
dnadesign/opal/
├─ src/                                     # OPAL code
├─ campaigns/
│   └─ <campaign_name>/
│       ├─ campaign.yaml                    # configuration (plugin refs + policies)
│       ├─ state.json                       # append-only campaign state across rounds
│       ├─ campaign.log.jsonl               # high-level event log
│       ├─ inputs/                          # add sequence:experimental label data here
│       ├─ outputs/
│       │   └─ round_<k>/
│       │       ├─ model.joblib             # frozen model (includes scaler state)
│       │       ├─ selection_top_k.csv      # summary of round results
│       │       ├─ ...
│       │       ├─ round_ctx.json           # RoundContext artifact
│       │       └─ round.log.jsonl          # fine-grained events for this round
│       └─ records.parquet                  # sequences, X, Y, and per-round outputs live here
└─ README.md
```

> **Note:** USR datasets typically live under `src/dnadesign/usr/datasets/<dataset>/records.parquet` and are **not copied** into campaigns.

---

## Configuration (`campaign.yaml`)

`campaign.yaml` declares **what to plug in** (transforms, model, objective, selection) and **how to run** (policies, sorting).

### Key blocks

* `campaign`: `name`, `slug`, `workdir`
* `data`: `location` (USR/local), `x_column_name`, `y_column_name`, `y_expected_length`
* `transforms_x`: `{ name, params }` (raw X → model-ready X)
* `transforms_y`: `{ name, params }` (tidy CSV → model-ready Y)
* `models`: `{ name, params }`
* `objectives`: `{ name, params }`
* `selection`: `{ name, params }`
* `training.target_scaler` and `scoring.sort_stability`

### Minimal example

```yaml
campaign: { name: My Campaign, slug: my_campaign, workdir: src/dnadesign/opal/campaigns/my_campaign }

data:
  location: { kind: local, path: campaigns/my_campaign/records.parquet }
  x_column_name: rep__vec
  y_column_name: y
  y_expected_length: 8

transforms_x: { name: identity,            params: {} }
transforms_y: { name: logic5_from_tidy_v1, params: {} }

models:
  name: random_forest
  params: { n_estimators: 100, random_state: 7 }

objectives:
  name: sfxi_v1
  params: { setpoint_vector: [0, 0, 0, 1] }

selection:
  name: top_n
  params: { top_k_default: 12, tie_handling: competition_rank }

training:
  target_scaler: { enable: true, minimum_labels_required: 5 }

scoring:
  score_batch_size: 10000
  sort_stability: "(-opal__{slug}__r{round}__selection_score__{objective}, id)"
```

> `{objective}` in `sort_stability` is replaced with your objective plugin name.

---

## Architecture & data flow

OPAL’s pipeline is strict and plugin-driven:

```
tidy.csv ──► transforms_y ──► labels[id,y]
records.parquet[X] ──► transforms_x ──► X
X + labels ──► model.fit
model + RoundContext ──► predict Ŷ ──► objective ──► score ──► selection
```

### RoundContext (per-round state)

`RoundContext` captures everything needed to **score** and **reproduce** a round without excessively adding columns to **records.parquet**.

**Typical fields (persisted to `outputs/round_<k>/round_ctx.json`):**

* **Identity & code**: `slug`, `round_index`, `run_id`, `code_version`
* **Config snapshot**: plugin names + params (transforms\_x/y, model, objective, selection), `y_expected_length`
* **Setpoint & label pool**: `setpoint`, `label_ids`, `training_label_count`
* **Objective scaling** (if applicable): `effect_pool_for_scaling`, `percentile_cfg` (e.g., `{p:95, fallback_p:75, min_n:5, eps:1e-8}`)
* **Determinism**: RNG seed(s)
* **Fingerprint**: short + full SHA-256 digest over the elements above

**Why it matters**

* Keeps per-row storage minimal while preserving all decisions behind a round
* Enables re-computing of round-specific objectives/scores/diagnostics offline
* Makes audits and “explain this selection” straightforward

---

## Data contracts

OPAL expects a well-formed records table and writes a set of columns per round.

### Records schema (USR essentials)

These columns are required (or created at ingest) in `records.parquet`:

| column       | type            | notes                  |
| ------------ | --------------- | ---------------------- |
| `id`         | string          | unique per record      |
| `bio_type`   | string          | `"dna"` or `"protein"` |
| `sequence`   | string          | case-insensitive       |
| `alphabet`   | string          | e.g. `dna_4`           |
| `length`     | int32           | `len(sequence)`        |
| `source`     | string          | provenance             |
| `created_at` | timestamp (UTC) | ingest time            |

**Representation (X):** Arrow `list<float>` or JSON array string; **fixed length** across used rows.
**Labels (Y):** Arrow `list<float>`; label history is append-only JSON in `opal__<campaign>__label_hist`.

**Namespacing rule:** secondary columns use `<tool>__<field>`.

### Per-round write-backs

For each round `k`, OPAL writes only the essentials:

* `opal__<campaign>__r<k>__pred_y`
* `opal__<campaign>__r<k>__selection_score__<objective>`
* `opal__<campaign>__r<k>__rank_competition`
* `opal__<campaign>__r<k>__selected_top_k_bool`
* `opal__<campaign>__r<k>__uncertainty__mean_all_std` *(optional)*
* `opal__<campaign>__label_hist`

> Richer, objective-specific diagnostics are often stored in **round artifacts** (`outputs/round_<k>/…`).

---

## Demo campaign

For a runnable demo and walkthrough, see the [Demo Guide](./campaigns/demo/README.md)

---

@e-south