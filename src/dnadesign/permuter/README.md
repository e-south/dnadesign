# **permuter**

*A bioinformatic pipeline for systematic permutation and **multi-metric** evaluation of biological sequences.*

---

## Overview

Given one or more reference sequences, **permuter** can

1. **Permute**

   * Nucleotide swapping – `scan_dna`
   * Codon swapping – `scan_codon` (requires a codon lookup table)

2. **Evaluate** each variant with pluggable **Evaluators** (per metric)

   * `log_likelihood` (LL)
   * `log_likelihood_ratio` (LLR) versus the reference
   * `embedding_distance` (negative Euclidean to the reference embedding; larger = better)

3. **Combine** metrics via an **objective** (e.g., `weighted_sum` with per-metric weights).

4. **Select** elites using a **strategy** (`top_k` or `threshold`).

5. **Iterate** permutation → evaluation → objective → selection for *N* rounds (optional).

6. **Report** JSONL (+ optional CSV) and any plots you request.

---

## Quick start

```bash
# 1) Put your references in CSV (two columns): ref_name,sequence
#    See: dnadesign/permuter/input/refs.csv

# 2) Run an example config (see full example below)
python -m dnadesign.permuter.main --config my_config.yaml
```

### “Plots-only” / analysis without recompute

If you already have JSONL outputs in `batch_results/<job>_<ref>/`:

```yaml
run:
  mode: analysis   # only (re)generate plots/CSVs from existing JSONL
```

Or let **permuter** decide automatically:

```yaml
run:
  mode: auto       # if outputs exist → plots-only; else → full pipeline
```

---

## Configuration

Define behavior in `config.yaml`.

```yaml
# dnadesign/permuter/config.yaml

permuter:
  experiment:
    name: demo

  jobs:
    - name: example_scan
      input_file: dnadesign/permuter/input/refs.csv
      references: ["retron_Eco1_msr_msd_wt"]

      # run mode
      run:
        mode: full                # one of: full | analysis | auto

      permute:
        protocol: scan_dna        # scan_dna | scan_codon | ...
        regions: []               # [] = full sequence; or [[start,end), …]
        lookup_tables: []         # e.g. ["dnadesign/permuter/input/codon_ecoli.csv"] for scan_codon

      evaluate:
        metrics:
          - id: ll
            name: log_likelihood
            evaluator: placeholder
            goal: max
            norm: {method: rank, scope: round}
          - id: llr
            name: log_likelihood_ratio
            evaluator: placeholder
            goal: max
            norm: {method: rank, scope: round}

      select:
        objective:
          type: weighted_sum
          weights: { ll: 0.7, llr: 0.3 }  # keys must match metric ids
        strategy:
          type: top_k
          k: 10
          include_ties: true

      iterate:
        enabled: true
        total_rounds: 3

      report:
        csv: true
        plots:
          - scatter_metric_by_position
```

#### Threshold strategy

You can threshold the **objective** or a specific **metric** (by id). Use either a **numeric threshold** or a **percentile**.

**Objective percentile (top 20% by objective):**

```yaml
select:
  objective:
    type: weighted_sum
    weights: { ll: 0.5, llr: 0.5 }
  strategy:
    type: threshold
    target: objective
    percentile: 80            # keep variants >= 80th percentile (within round)
```

**Metric threshold (normalized metric id `llr`, keep >= 0.8):**

```yaml
select:
  objective:
    type: weighted_sum
    weights: { ll: 0.5, llr: 0.5 }
  strategy:
    type: threshold
    target: metric
    metric_id: llr
    use_normalized: true      # default; keeps 0..1, goal-aware
    threshold: 0.8
```

> If you set `target: metric` and use `percentile`, you must use `use_normalized: true` (default).


---

#### Directory layout

```
dnadesign/permuter/
├── main.py                        # single entry-point
├── config.py                      # YAML validator
├── logging_utils.py
├── evaluate.py                    # multi-metric evaluation helpers
├── permute_record.py
├── reporter.py                    # JSONL/CSV + dynamic plotting
│
├── protocols/                     # mutation strategies
│   ├── __init__.py
│   ├── scan_dna.py
│   └── scan_codon.py              # (if present) requires lookup table
│
├── evaluators/                    # scoring back ends
│   ├── __init__.py
│   ├── base.py
│   └── placeholder.py             # deterministic stub (no GPU)
│
├── selector/                      # 🔌 selection plugins
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
├── plots/                         # 🔌 drop-in visualisations
│   └── scatter_metric_by_position.py
│
└── input/                         # user data
    ├── refs.csv                   # required columns: ref_name,sequence
    └── codon_ecoli.csv            # example codon table
```

Outputs land in `batch_results/<job>_<ref>/`.

---

## Output files

```
batch_results/lacI_scan_lacI/
├── MANIFEST.json
├── config_snapshot.yaml
├── r1_variants.jsonl
├── r1_elites.jsonl
├── norm_stats_r1.json
├── r2_variants.jsonl
├── r2_elites.jsonl
├── norm_stats_r2.json
├── plots/
│   └── lacI_scan_scatter_metric_by_position.png
├── lacI_scan.csv                  # optional (report.csv: true)
└── lacI_scan_elites.csv           # optional (report.csv: true)
```

### JSONL record (example)

```json
{
  "var_id": "FQ2FZ8S1L1GX",
  "parent_var_id": "9K7U5Z4YB2QC",
  "job_name": "lacI_scan",
  "ref_name": "lacI",
  "protocol": "scan_dna",
  "round": 2,
  "sequence": "ACGT…",
  "modifications": ["A5T","G12C"],
  "metrics": {"llr": 0.14, "emb": -0.82},
  "norm_metrics": {"llr": 0.73, "emb": 0.58},
  "objective_score": 0.684,
  "objective_meta": {
    "type": "weighted_sum",
    "weights": {"llr": 0.7, "emb": 0.3},
    "norm_scope": "round",
    "norm_stats_id": "r2_9A3F10"
  }
}
```

---

## Extending **permuter**

| Want to add …           | Where / how                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| **Mutation protocol**   | Create `permuter/protocols/my_protocol.py` with `generate_variants()`                       |
| **Evaluator / model**   | Subclass `evaluators.base.Evaluator`, register in `evaluators/__init__.py`                  |
| **Objective (combine)** | Add under `selector/objectives/`, subclass `Objective`, register in `__init__.py`           |
| **Strategy (select)**   | Add under `selector/strategies/`, subclass `Strategy`, register in `__init__.py`            |
| **Plot**                | Drop a module in `permuter/plots/` exposing `plot(elite_df, all_df, output_path, job_name)` |

---

Eric J. South (ericjohnsouth@gmail.com)
