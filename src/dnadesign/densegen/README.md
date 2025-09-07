# DenseGen — Dense Array Generator

**DenseGen** is a DNA sequence design pipeline for **batch assembly of synthetic bacterial promoters** composed of **densely packed transcription factor binding sites (TFBSs)**. It wraps the [`dense-arrays`](https://github.com/e-south/dense-arrays) ILP solver.

- **Add TFBS data:** Provide a CSV of TF→TFBS pairs or seed from a USR dataset.
- **Add explicit constraints:** Use YAML to define **per-constraint output quotas** and **fixed elements** (e.g., σ70 promoter motifs).
- **Generate dense arrays:** Build a motif library sized to your target length, run the ILP, and avoid stalling via resampling + duplicate guards.
- **Flexible outputs:** Write to **USR** (default) or **JSONL** (or **both**) with identical metadata.

### Pipeline overview

1. **Inputs**
  - **CSV TF/TFBS**: two columns `tf`, `tfbs`. Place under `densegen/inputs/` and reference in YAML.
  - **USR sequences**: import sequences from an existing USR dataset.

2. **Sampling**
  - A motif library is drawn until the sum of lengths ≳ `sequence_length + subsample_over_length_budget_by`.
  - **TF coverage (optional):** `cover_all_tfs: true` ensures ≥1 TFBS per unique TF in the CSV (diversity across TFs).
  - **Binding-site diversity (optional):** `unique_binding_sites: true` forbids duplicate TFBS strings in the same library.
  - **Tuning:** Larger `subsample_over_length_budget_by` ⇒ more options (potentially denser packing) but longer solves.

3. **Optimization**
  - Uses `dense-arrays` to generate dense arrays.
  - `solver.diverse_solution: true` (if supported) biases away from recent motif choices **within a single solver instance**.

4. **Outputs**

  - **USR (default):** sequences are imported with de-dup and namespaced metadata (`densegen__*`).
  - **JSONL:** line-delimited records with the same essential + derived fields (resume-safe via deterministic IDs).

### Directory layout

```
dnadesign/
└─ densegen/
    ├─ src/                      # code
    ├─ config.yaml               # edit me
    ├─ inputs/                   # CSVs (e.g., tf2tfbs_mapping.csv)
    └─ outputs/                  # JSONL output (if enabled)
```

### Minimal CSV example

`inputs/tf2tfbs_mapping.csv`
```csv
tf,tfbs
LexA,TACTGTATATATACAGTA
LexA,CTGTATATACAGTATACG
LexA,ATACAGTATACGTTACAT
LexA,GGTTACATATGTACAGTA
LexA,TATACAGTATACGATGTA
CpxR,GTAAACCAATTGTTTAC
CpxR,ACCAATTGTTTACGGTA
CpxR,TTGTTTACGGTAAACCA
CpxR,AACCAATTGTTTACGTA
CpxR,CAATTGTTTACGTAAAC
```

### Configuration reference (key fields)

```yaml
densegen:
  inputs:
    - name: 60bp_dual_promoter_cpxR_LexA
      type: csv_tfbs
      path: inputs/tf2tfbs_mapping_cpxR_LexA.csv

  output:
    kind: usr         # usr | jsonl | both
    jsonl:
      path: outputs/60bp_dual_promoter_cpxR_LexA.jsonl

  usr:
    dataset: 60bp_dual_promoter_cpxR_Lex
    root: null
    namespace: densegen
    chunk_size: 128
    allow_overwrite: true

  generation:
    sequence_length: 60
    quota: 30000
    sampling:
      subsample_over_length_budget_by: 120
      cover_all_tfs: true            # ensure ≥1 TFBS per unique TF in the CSV
      unique_binding_sites: true     # forbid duplicate TFBS strings in a library
      max_sites_per_tf: null         # cap per TF after coverage (null = no cap)
      relax_on_exhaustion: true      # if we can't meet budget under caps, relax

    plan:
      - name: sigma70_high
        quota: 10000
        fixed_elements:
          promoter_constraints:
            upstream: "TTGACA"
            downstream: "TATAAT"
            spacer_length: [16, 18]
            upstream_pos: [0, 60]
      - name: sigma70_mid
        quota: 10000
        fixed_elements:
          promoter_constraints:
            upstream: "ACCGCG"
            downstream: "TATAAT"
            spacer_length: [16, 18]
            upstream_pos: [0, 60]
      - name: sigma70_low
        quota: 10000
        fixed_elements:
          promoter_constraints:
            upstream: "GCAGGT"
            downstream: "TATAAT"
            spacer_length: [16, 18]
            upstream_pos: [0, 60]

  solver:
    backend: CBC
    diverse_solution: true
    options:
      - "Threads=16"
      - "TimeLimit=10"

  runtime:
    round_robin: true
    arrays_generated_before_resample: 20
    # Enforce that the *final solution* must include at least N sites per TF
    # present in the sampled library (post-solve check).
    require_min_count_per_tf: true
    # min_count_per_tf: 1             # optional explicit integer (overrides boolean)
    max_duplicate_solutions: 3        # dup-guard (consecutive)
    stall_seconds_before_resample: 30 # stall-guard (no solution within this many seconds)
    stall_warning_every_seconds: 15
    max_resample_attempts: 3          # see semantics below
    random_seed: 42

  postprocess:
    fill_gap: true
    fill_gap_end: "5prime"            # or "3prime"
    fill_gc_min: 0.40
    fill_gc_max: 0.60

  logging:
    level: INFO
    suppress_solver_stderr: true
    print_visual: true
```

**Resampling is triggered by:**

* completing a subsample (i.e., after producing `arrays_generated_before_resample` sequences),
* stall guard (`stall_seconds_before_resample`) if **no** solution appears in that time, or
* duplicate guard (`max_duplicate_solutions`) when we see that many identical sequences in a row from the solver.

**`max_resample_attempts` semantics (applies in RR and non-RR):**

* Interpreted **within a single “subsample try”** to reach `arrays_generated_before_resample`.
* If we can’t hit that local target after at most `max_resample_attempts` resamples, we **move on**:

  * **RR:** return to the scheduler (other items get a turn).
  * **Non-RR:** start a fresh try toward the global quota (we don’t kill the run).

## Output schema

All derived fields are stored **namespaced** as `densegen__<key>` in USR and exactly as keys in JSONL (with `densegen__` prefix already applied by the JSONL sink). Below is the logical schema (keys shown un-namespaced for readability):

### Essentials

* `sequence` — the designed sequence (post gap-fill if enabled)
* `bio_type` — `"dna"`
* `alphabet` — `"dna_4"`
* `source` — `"densegen:<input_name>:<plan_name>"`
* `id` (JSONL only) — deterministic SHA256 of `(bio_type|alphabet|sequence)`

### Derived (metadata)

* `plan` — name of the plan item (e.g., `sigma70_high`)
* `sequence_length` — configured target length
* `library_size` — motifs in the library shown to the solver
* `solver` — backend, e.g., `CBC`; `diverse` — whether diverse solutions were requested
* `visual` — ASCII visual of the dense placement (forward & reverse)
* `compression_ratio` — total motif length in solution / sequence\_length
* `promoter_constraint` — name (if provided) of the promoter constraint used
* `gap_fill_used`, `gap_fill_bases`, `gap_fill_end`, `gap_fill_gc_min`, `gap_fill_gc_max` * gap-fill details if padding was added
* `tf_list` — **unique** TF names present in the sampled library for this subsample
* `tfbs_parts` — one-to-one with the library order: `"TF:TFBS"` strings
* `used_tfbs` — **what actually made it into the final sequence**, one entry per placement (`"TF:TFBS"`, duplicates allowed)
* `used_tfbs_detail` — list of objects `{tf, tfbs, orientation, offset}` for each placement actually used
* `used_tf_counts` — map of TF→count in the final sequence (e.g., `{"LexA": 2, "CpxR": 1}`)
* `used_tf_list` — sorted list of TFs actually present in the final sequence
* `covers_all_tfs_in_solution` — `true/false` if the **post-solve** check passed
* `min_count_per_tf_required` — the requirement used (0 means disabled)

### Tips

* If you want stricter per-TF representation in the **final sequence**, set:

  * `runtime.require_min_count_per_tf: true` (defaults to 1), or
  * `runtime.min_count_per_tf: N` for N≥1.
    Solutions violating this are rejected (the solver is asked not to repeat them), and sampling continues.

@e-south
