## permuter

permuter is a bioinformatics pipeline for biological sequence permutation and subsequent evaluation workflows.

#### Overview

Given a reference sequence, permuter performs:

1. **Permutation** (point mutations or codon swaps) under `scan_dna` or `scan_codon` protocols.
2. **Evaluation** via an external tool (e.g. Evo2 forward pass) to compute:

   * **Log­likelihood** (sum of token log‑probs)
   * **Log­likelihood ratio (LLR)** vs. reference
   * **Embedding / Euclidean distance** in latent space
3. **Selection** of top variants (`top_k` or threshold).
4. **Iteration** of permute→evaluate→select for N rounds (optional).
5. **Reporting** of final elites as JSON Lines and plots.

#### Directory Structure

```
permuter/
├── config.yaml           # experiment & job definitions
├── main.py               # orchestrates pipeline
├── permute_record.py     # generator yielding one variant at a time
├── protocols/            # mutation rules
│   ├── scan_dna.py       # nucleotide‐level variants
│   └── scan_codon.py     # codon‐level swaps (uses frequency lookup tables)
├── evaluator.py          # scoring logic
├── selector.py           # top_k or threshold selection
├── iterator.py           # multi‐round loop
└── reporter.py           # writes JSON & generates plots
```

#### Configuration (`config.yaml`)

Define jobs under `permuter:`. Example:

```yaml
permuter:
  experiment:
    name: my_experiment
  jobs:
    - name: example_scan
      input_file: refs.csv
      references: ["my_ref_sequence_name"]
      protocol: scan_dna         # or scan_codon
      permute:
        regions: []             # [] = full sequence; or [[start,end),…]
        lookup_tables: []       # e.g. ["data/codon_ecoli.csv"]
      evaluate:
        evaluator: evo2_7b
        metric: log_likelihood_ratio
      select:
        strategy: top_k
        k: 10
      iterate:
        enabled: false
        total_rounds: 3
      report:
        plots: true
```

#### Usage

```bash
python main.py --config config.yaml
```

#### Outputs

Results for each job go to `results/{job_name}/`:

* **`elites.jsonl`**: JSON Lines of selected variants. Example entry:

  ```json
  {
    "id": "<uuid>",
    "sequence": "ATGCCG…",
    "protocol": "scan_dna",
    "ref_name": "my_ref_sequence_name",
    "modifications": [{"pos":5,"from":"A","to":"G"}],
    "score": 0.456,
    "score_type": "llr",
    "round": 1,
  }
  ```

* *(Optional)* plots of variants and their corresponding metrics

#### Extending Permuter

* **Add protocols**: drop new sequence permutation modules (`*.py`) into `protocols/` and set `protocol:` in config.
* **Custom codon tables**: reference CSV paths in `lookup_tables`.
* **Evaluator**: adjust logic in `evaluator.py`.
* **Selectors**: adjust logic in `selector.py`.

---

*Author: Eric J. South (ericjohnsouth@gmail.com)*
