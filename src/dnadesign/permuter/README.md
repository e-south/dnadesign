## **permuter**

_A bioinformatic pipeline for systematic permutation and evaluation of biological sequences._


---

#### Overview

Given one or more reference sequences, **permuter** can

1. **Permute**  
   * Nucleotide saturation – `scan_dna`  
   * Codon scanning – `scan_codon`

2. **Evaluate** each variant with a pluggable **Evaluator**  
   * Log-likelihood (LL)  
   * Log-likelihood ratio (LLR) versus the reference  
   * Embedding distance (negative Euclidean; larger = better)

3. **Select** elites (`top_k` or `threshold`).

4. **Iterate** permutation → evaluation → selection for _N_ rounds (optional).

5. **Report** tidy CSV/JSONL outputs **and** any plots you request.

---

#### Quick start

```bash
# 1. Run an example config
python -m permuter.main --config my_config.yaml
````

A dry-run requires **no GPU / heavy model**:

```yaml
evaluate:
  evaluator: placeholder   # deterministic stub
  metric: log_likelihood
```

The default scatter plot is still produced.

---

#### Directory layout

```
permuter/
├── main.py                    # single entry-point
├── config.py                  # YAML validator
├── logging_utils.py
├── permute_record.py
├── iterator.py
├── selector.py
├── reporter.py                # tables + dynamic plotting
│
├── protocols/                 # mutation strategies
│   ├── scan_dna.py
│   └── scan_codon.py
│
├── evaluators/                # scoring back ends
│   ├── base.py
│   ├── placeholder.py
│   └── evo2.py                # (optional) Evo-2 wrapper
│
├── plots/                     # 🔌 drop-in visualisations
│   └── scatter_metric_by_position.py
│
└── input/                     # user data
    ├── refs.csv               # required columns: ref_name,sequence
    └── codon_ecoli.csv        # example codon table
```

Outputs land in `batch_results/YYYY-MM-DD/<job>/<ref>/`.

---

#### Configuration (`config.yaml`)

```yaml
permuter:
  experiment:
    name: demo
  jobs:
    - name: lacI_scan
      input_file: input/refs.csv
      references: ["lacI"]
      protocol: scan_dna

      permute:
        regions: []                 # [] = full sequence; or [[start,end),…]
        lookup_tables: []           # e.g. ["data/codon_ecoli.csv"]

      evaluate:
        evaluator: placeholder      # or evo2_7b
        metric: log_likelihood_ratio

      select:
        strategy: top_k
        k: 50

      iterate:
        enabled: true
        total_rounds: 3

      report:
        plots:                      # pick any modules under permuter.plots
          - scatter_metric_by_position
```

#### Validation rules (asserted at start-up)

| Field                | Rule (abridged)                                   |
| -------------------- | ------------------------------------------------- |
| `references[]`       | must exist in `refs.csv`                          |
| `permute.regions`    | `0 ≤ start < end ≤ len(seq)`                      |
| `scan_codon`         | requires at least one codon table                 |
| `select.strategy`    | must supply **only** the args it needs            |
| `embedding_distance` | supply exactly one `embedding_reference_sequence` |

Mis-configurations raise **ConfigError**.

---

#### Plots

Add your own:

```python
# permuter/plots/heatmap_llr.py
def plot(elite_df, all_df, output_path, job_name):
    ...
```

Then list it:

```yaml
report:
  plots: ["scatter_metric_by_position", "heatmap_llr"]
```

---

####  Extending permuter

| Want to add …         | Where / how                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| **Mutation protocol** | Create `permuter/protocols/my_protocol.py` with `generate_variants()`       |
| **Evaluator / model** | Sub-class `evaluators.base.Evaluator`, register in `evaluators/__init__.py` |
| **Plot**              | Drop a module in `permuter/plots/` exposing `plot()`                        |
| **Selector**          | Tweak logic in `selector.py` (it’s \~25 lines)                              |


---

#### Output files

```
batch_results/2025-08-07/lacI_scan/lacI/
├── lacI_scan.csv              # all variants (tidy)
├── lacI_scan_elites.csv       # elite subset
├── lacI_scan_scatter_metric_by_position.png
└── ... more plots as requested
```

---

Eric J. South (ericjohnsouth@gmail.com)

