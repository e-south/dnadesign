# DenseGen — Dense Array Generator

**DenseGen** designs compact synthetic promoters by packing **transcription factor binding sites (TFBSs)** into fixed-length sequences. It wraps the [`dense-arrays`](https://github.com/e-south/dense-arrays) ILP optimizer and adds sampling, constraints, batching, outputs, and plotting.

- **Inputs:** A CSV of `tf,tfbs` pairs (or sequences from a USR dataset)
- **Constraints:** Optional promoter motifs (e.g., σ70 `TTGACA … TATAAT`) and placement rules
- **Robustness:** Library resampling, duplicate/stall guards, and resume‑safe outputs
- **Outputs:** USR dataset (default) and/or JSONL with consistent, namespaced metadata
- **Visualization:** Built‑in plots for usage, coverage, compression, and more

---

## Quick start

> **Prerequisites:** Python, [`dense-arrays`](https://github.com/e-south/dense-arrays), and a MILP solver (e.g., **CBC** or **GUROBI**). CBC is open-source; GUROBI is supported if installed/licensed.

1) **Create a TF/TFBS CSV**

`densegen/inputs/tf2tfbs_mapping.csv`
```csv
tf,tfbs
LexA,TACTGTATATATACAGTA
LexA,CTGTATATACAGTATACG
CpxR,GTAAACCAATTGTTTAC
CpxR,ACCAATTGTTTACGGTA
````

2. **Edit the job YAML**

`densegen/jobs/config.yaml` (minimal)

```yaml
densegen:
  inputs:
    - name: demo
      type: csv_tfbs
      path: inputs/tf2tfbs_mapping.csv

  output:
    kind: usr            # usr | jsonl | both
    jsonl:
      path: outputs/demo.jsonl

  usr:
    dataset: demo_densegen
    namespace: densegen   # namespacing for attached metadata

  generation:
    sequence_length: 60
    quota: 200
    sampling:
      subsample_over_length_budget_by: 120
      cover_all_tfs: true
      unique_binding_sites: true

    plan:
      - name: sigma70
        quota: 200
        fixed_elements:
          promoter_constraints:
            upstream: "TTGACA"
            downstream: "TATAAT"
            spacer_length: [16, 18]
            upstream_pos: [0, 60]

  solver:
    backend: CBC
    diverse_solution: true
    options:
      - "Threads=8"
      - "TimeLimit=10"

  runtime:
    round_robin: true
    arrays_generated_before_resample: 20
    require_min_count_per_tf: true
    max_duplicate_solutions: 3
    stall_seconds_before_resample: 30
    max_resample_attempts: 3
    random_seed: 42

  postprocess:
    fill_gap: true
    fill_gap_end: "5prime"
    fill_gc_min: 0.40
    fill_gc_max: 0.60

  logging:
    level: INFO
    suppress_solver_stderr: true
    print_visual: true
```

3. **Run the pipeline**

If you installed a console script, the equivalent commands are:

```bash
dense validate -c densegen/jobs/config.yaml
dense plan     -c densegen/jobs/config.yaml
dense run      -c densegen/jobs/config.yaml
dense plot     -c densegen/jobs/config.yaml --only tf_usage,tf_coverage
```

---

## How it works

1. **Sampling**
   From your TF/TFBS CSV, DenseGen builds a **motif library** big enough to cover
   `sequence_length + subsample_over_length_budget_by`. You can require ≥1 site for each TF (`cover_all_tfs`) and/or forbid duplicate TFBS strings in the same library (`unique_binding_sites`). If a per‑TF cap is set (`max_sites_per_tf`) and the budget can’t be reached, it is relaxed (when `relax_on_exhaustion: true`).

2. **Optimization**
   The `dense-arrays` optimizer places library motifs densely into the target length while honoring **promoter constraints** (e.g., σ70 `upstream/downstream` motifs, `spacer_length`, and optional motif positions). `diverse_solution: true` requests variety across solutions from the same optimizer instance.

3. **Runtime guards**

   * **Duplicate guard:** resample if `max_duplicate_solutions` identical sequences appear consecutively.
   * **Stall guard:** resample if no solution is produced within `stall_seconds_before_resample`.
   * **Coverage check:** if `require_min_count_per_tf` (or `min_count_per_tf: N`) is enabled, any solution failing this post‑solve rule is discarded.

4. **Post‑processing**
   If the packed motifs underfill the sequence, DenseGen **gap‑fills** with random bases while targeting a GC window (`fill_gc_min/max`) on the chosen end.

5. **Outputs**

   * **USR** (default): sequences imported with namespaced metadata (`densegen__*`).
   * **JSONL** (optional): one record per line with the same fields (IDs are deterministic SHA256 of `bio_type|alphabet|sequence`).

---

## Configuration cheat‑sheet

* **Inputs**

  * `type: csv_tfbs` → CSV with columns `tf, tfbs`
  * `type: usr_sequences` → reuse sequences from an existing USR dataset
    *Relative paths are resolved against the DenseGen tree first, then project root, then `src/`.*

* **Generation**

  * `sequence_length` – target promoter length
  * `quota` – total number of sequences across all plan items
  * `sampling.subsample_over_length_budget_by` – extra library length to reach before solving (more options → potentially denser packing)
  * `plan` – one or more constraint “buckets,” each with `name`, `quota|fraction`, and optional `fixed_elements.promoter_constraints`:

    * `upstream`, `downstream` (motif strings or `"none"`)
    * `spacer_length: [min, max]`
    * `upstream_pos`, `downstream_pos`: `[start, end]` allowed ranges (optional)

* **Solver**

  * `backend` – `CBC` or `GUROBI` (probed once at startup)
  * `diverse_solution` – request variety within a run
  * `options` – passed through to the solver (e.g., `Threads=16`, `TimeLimit=10`)

* **Runtime**

  * `arrays_generated_before_resample` – how many sequences to emit per subsample before resampling the library
  * `require_min_count_per_tf` / `min_count_per_tf` – enforce ≥N placements per TF **in the final sequence**
  * `max_duplicate_solutions`, `stall_seconds_before_resample`, `max_resample_attempts`
  * `random_seed` – global RNG seed

* **Postprocess**

  * `fill_gap: true|false`, `fill_gap_end: 5prime|3prime`, `fill_gc_min/max`

* **Logging**

  * `level`, `suppress_solver_stderr`, `print_visual` (prints the ASCII placement map)

---

## Output fields (most useful)

DenseGen stores metadata **namespaced** as `densegen__<key>`.

**Essentials**

* `sequence`, `bio_type="dna"`, `alphabet="dna_4"`, `source`
* `id` (JSONL only; deterministic)

**Derived highlights**

* `plan`, `sequence_length`, `library_size`
* `solver`, `diverse`, `compression_ratio`
* `promoter_constraint` (name if provided)
* `used_tfbs` (list of `"TF:TFBS"` used in the final sequence)
* `used_tfbs_detail` (list of `{tf, tfbs, orientation, offset}`)
* `used_tf_counts` (e.g., `{"LexA": 2, "CpxR": 1}`)
* Gap‑fill info: `gap_fill_used`, `gap_fill_bases`, `gap_fill_end`, `gap_fill_gc_min/max`, `gap_fill_gc_actual`
* Coverage gate: `covers_all_tfs_in_solution`, `min_count_per_tf_required`

---

## Plotting

Generate plots from the YAML with:

```bash
dense plot -c densegen/jobs/config.yaml
# or select a subset:
dense plot -c ... --only tf_usage,tf_coverage
```

**Available plots**

* `compression_ratio` – histogram of compression ratios
* `tf_usage` – total counts per TF (across all sequences)
* `tfbs_usage` – ranked counts of individual TFBS strings (bars colored by TF)
* `plan_counts` – sequences per plan item
* `tf_coverage` – per‑position coverage along the sequence (supports `fill`, `stacked`, smoothing)
* `tfbs_length_density` – per‑TF distribution of TFBS lengths
* `gap_fill_gc` – scatter of gap‑fill GC fraction vs gap bases (filled only)

Output PDFs are written to `plots.out_dir` (default: `densegen/outputs/plots`).

---

## Tips & troubleshooting

* **Solver probe failed**
  Set `solver.backend` to an installed solver (e.g., `CBC`) or install/configure your preferred solver (e.g., GUROBI).

* **Few or no solutions**
  Increase `sampling.subsample_over_length_budget_by`; relax `unique_binding_sites` or per‑TF caps; check promoter constraints; shorten `sequence_length`.

* **Stalling**
  See `runtime.stall_seconds_before_resample` and `max_resample_attempts`. Duplicate sequences trigger resampling via `max_duplicate_solutions`.

* **Reproducibility**
  Use `runtime.random_seed`. For exact reruns, keep the same YAML and input CSV.

---

## Project layout

```
dnadesign/
└─ densegen/
   ├─ jobs/               # job YAMLs
   ├─ inputs/             # TF/TFBS CSVs (your data)
   ├─ outputs/            # JSONL (optional) and plots
   └─ src/dnadesign/densegen/src/
       ├─ main.py         # CLI
       ├─ data_ingestor.py, sampler.py
       ├─ optimizer_wrapper.py
       ├─ outputs.py, usr_adapter.py
       └─ plotting.py
```

---

@e-south
