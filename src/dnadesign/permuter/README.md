## permuter

*A bioinformatic pipeline for systematic permutation and **multi-metric** evaluation of biological sequences.*


---

### Table of contents

* [Overview](#overview)
* [Install](#install)
* [Quick start](#quick-start)
* [Configuration](#configuration)

  * [workspace.yaml](#workspaceyaml)
  * [Experiment config.yaml](#experiment-configyaml)
  * [Run modes](#run-modes)
  * [Selection strategies](#selection-strategies)
* [Outputs](#outputs)
* [Directory layout](#directory-layout)
* [Extending permuter](#extending-permuter)

---

## Overview

Given one or more reference sequences, **permuter**:

1. **Permutes** sequences via a protocol

   * `scan_dna` — nucleotide substitutions
   * `scan_codon` — codon substitutions (requires a codon lookup table)

2. **Evaluates** each variant using pluggable **Evaluators** (one per metric)
   Examples in tree:

   * `log_likelihood` (LL)
   * `log_likelihood_ratio` (LLR) vs reference
   * `embedding_distance` (negative Euclidean to reference embedding)

3. **Combines** metrics with an **objective** (e.g., `weighted_sum` with per-metric weights)

4. **Selects** elites via a **strategy** (`top_k` or `threshold`)

5. **Iterates** (optional): permutation → evaluation → objective → selection for N rounds

6. **Reports** JSONL (+ optional CSV) and dynamic plots you specify

---

## Install

```bash
# From the project root (editable install recommended during development)
pip install -e .
```

---

### Quick start

The workspace lets you manage multiple experiments cleanly, each with its own `config.yaml` and inputs.

1. Ensure a workspace file exists:

```
dnadesign/permuter/workspace.yaml
```

Example (see full schema below):

```yaml
permuter:
  workspace:
    experiments_dir: dnadesign/permuter/experiments
    runs:
      - name: example_nt_scan
        config: config.yaml
        enabled: true
```

2. Create the experiment folder:

```
dnadesign/permuter/experiments/example_nt_scan/
  ├─ config.yaml
  └─ input/
     └─ refs.csv   # columns: ref_name,sequence
```

3. Run:

```bash
# list experiments
python -m dnadesign.permuter.main --config dnadesign/permuter/workspace.yaml --list

# run all enabled experiments
python -m dnadesign.permuter.main --config dnadesign/permuter/workspace.yaml

# run a subset by name
python -m dnadesign.permuter.main --config dnadesign/permuter/workspace.yaml --only example_nt_scan

```

---

### Configuration

### `workspace.yaml`

Located at the repo root. It simply enumerates experiments and where to find their configs.

```yaml
permuter:
  workspace:
    # Folder that contains subfolders per experiment:
    experiments_dir: dnadesign/permuter/experiments

    runs:
      - name: example_nt_scan     # must be unique
        config: config.yaml       # path relative to experiments_dir/<name>
        enabled: true             # included when running without --only
      # - name: another_experiment
      #   config: config.yaml
      #   enabled: false
```

### Experiment `config.yaml`

Each experiment is self-contained. Minimal example:

```yaml
permuter:
  experiment:
    name: permuter_demo

  jobs:
    - name: example_nt_scan
      input_file: input/refs.csv                 # must include ref_name, sequence
      references: ["retron_Eco1_msr_msd_wt"]

      run:
        mode: auto                               # full | analysis | auto

      permute:
        protocol: scan_dna                       # scan_dna | scan_codon
        regions: []                              # [] = full sequence; or [[start,end), …]
        lookup_tables: []                        # required for scan_codon

      evaluate:
        metrics:
          - id: llr
            name: log_likelihood_ratio
            evaluator: evo2_llr
            goal: max
            params:
              model_id: evo2_7b
              device: cuda:0
              precision: bf16
              alphabet: dna
              reduction: mean

      select:
        objective:
          type: weighted_sum
          weights: { llr: 1 }                     # keys must match metric ids exactly
        strategy:
          type: top_k
          k: 10
          include_ties: true

      iterate:
        enabled: true
        total_rounds: 3

      report:
        csv: false
        plots:
          - position_scatter_and_heatmap
          - metric_by_mutation_count
```


### Run modes

```yaml
run:
  mode: full      # do all rounds, write JSONL/plots
  # or
  mode: analysis  # ONLY (re)generate plots/CSVs from existing JSONL in batch_results
  # or
  mode: auto      # if outputs exist → behave like analysis; else → full
```

### Selection strategies

**Top-K**

```yaml
select:
  objective:
    type: weighted_sum
    weights: { ll: 1 }
  strategy:
    type: top_k
    k: 10
    include_ties: true
```

**Threshold (objective percentile)**

```yaml
select:
  objective:
    type: weighted_sum
    weights: { ll: 1 }
  strategy:
    type: threshold
    target: objective
    percentile: 80        # keep variants >= 80th percentile within the round
```

**Threshold (metric by id)**

```yaml
select:
  objective:
    type: weighted_sum
    weights: { ll: 1 }
  strategy:
    type: threshold
    target: metric
    metric_id: ll
    use_normalized: true  # required when using percentile with target=metric
    threshold: 0.8
```

---

### Outputs

Each job×reference pair writes to:

```
dnadesign/permuter/experiments/<exp>/batch_results/<job>_<ref>/
├── MANIFEST.json
├── config_snapshot.yaml
├── r1_variants.jsonl
├── r1_elites.jsonl
├── r2_variants.jsonl
├── r2_elites.jsonl
├── ...
├── plots/
│   ├── <metric>_<ref>_position_scatter_and_heatmap.png
│   └── <metric>_<ref>_metric_by_mutation_count.png
├── <job>.csv            # optional (report.csv: true)
└── <job>_elites.csv     # optional
```

**JSONL** is the single source of truth. The MANIFEST tracks rounds, plot paths, and reference metadata to support “analysis” reruns without recomputation.

---

### Directory layout

```
dnadesign/permuter/
├── main.py                          # thin CLI entry; detects single vs workspace
├── runner.py                        # experiment/job orchestration
├── workspace.py                     # workspace.yaml loader
├── config.py                        # config validator
├── logging_utils.py
├── evaluate.py
├── permute_record.py                # protocol dispatch (decoupled)
├── reporter.py                      # JSONL/CSV + dynamic plotting w/ subtitles
│
├── experiments/
│   └── example_nt_scan/
│       ├── config.yaml
│       └── input/                   # experiment-local inputs live here
│           └── refs.csv
│
├── protocols/
│   ├── __init__.py
│   ├── scan_dna.py
│   └── scan_codon.py
│
├── evaluators/
│   ├── __init__.py
│   ├── base.py
│   └── placeholder.py
│
├── selector/
│   ├── __init__.py
│   ├── objectives/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── weighted_sum.py
│   └── strategies/
│       ├── __init__.py
│       ├── base.py
│       ├── top_k.py
│       └── threshold.py
│
└── plots/
    ├── position_scatter_and_heatmap.py
    └── metric_by_mutation_count.py
```

---

### Extending **permuter**

| Add…                  | How                                                                                                                 |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Mutation protocol** | Create `protocols/my_protocol.py` with `generate_variants(ref_entry, params, regions, lookup_tables) -> List[Dict]` |
| **Evaluator / model** | Implement `evaluators.base.Evaluator.score()`, register in `evaluators/__init__.py`                                 |
| **Objective**         | Add under `selector/objectives/`, subclass `Objective`, register in `__init__.py`                                   |
| **Strategy**          | Add under `selector/strategies/`, subclass `Strategy`, register in `__init__.py`                                    |
| **Plot**              | Drop a module in `plots/` exposing `plot(elite_df, all_df, output_path, job_name, ...)`                             |

Plots receive a compact evaluator subtitle automatically (e.g., `"placeholder"` or `"eva + evb"`).


---

**Author:** Eric J. South (ericjohnsouth@gmail.com)
