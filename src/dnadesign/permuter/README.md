## Permuter

Mutate biological sequences, score them with pluggable evaluators, and export/plot the results — via a clean Typer CLI.

- Run in‑silico deep mutational scans.
- Keep storage simple: one Parquet file per run (plus plots).
- Be extensible: protocols/evaluators/plots are modular.

---

### Install

Permuter lives inside the dnadesign repo.
```bash
# cd dnadesign/
source .venv/bin/activate
```

---

### Project layout

```
src/dnadesign/permuter/
  src/         # source code lives here
  jobs/        # job YAMLs (define a job here)
  inputs/      # default inputs (refs.csv, tables, ...)
  runs/        # default output root (datasets written here)
```

* Paths in YAML are relative to the job YAML.
* `${JOB_DIR}`, env vars, and `~` are resolved.
* Dataset directory (`records.parquet`, `REF.fa`, `plots/`) is `<job.output.dir>/<ref_name>/` unless `--out` is given.

---

### Quick start

Place your DNA references in `inputs/refs.csv`.

Example:

```csv
 ref_name,sequence
 BL21_RNase_H1_wt,ACGTTG...TT     # full coding DNA sequence
```

**A) Run a nucleotide scan (single‑base)**

```bash
permuter run --job src/dnadesign/permuter/jobs/nt_scan_demo.yaml --ref BL21_RNase_H1_wt
# dataset → src/dnadesign/permuter/runs/nt_scan_demo/BL21_RNase_H1_wt/
```

**B) Run a codon scan (codon swaps)**

We include an example `codon_ecoli.csv` (subset shown):

````csv
codon,amino_acid,fraction,frequency
AAA,K,0.73,33.2
AAC,N,0.53,24.4
AAG,K,0.27,12.1
AAT,N,0.47,21.9
CAA,Q,0.30,12.1
CAC,H,0.45,13.1
CAG,Q,0.70,27.7
TAG,*,-,-
...
```

How substitutions are chosen: for each scanned codon position, `permuter` ranks codons by the provided weight and, for every other amino acid, switches the wild‑type codon to that amino acid’s **most frequent codon**.

**Example jobs:**

Run a deep mutational scan on nucleotides:
```yaml
job:
 name: rnaseh1_nt_scan
 bio_type: dna
 input:
   refs: "${JOB_DIR}/../inputs/refs.csv"
   name_col: "ref_name"
   seq_col: "sequence"
 permute:
   protocol: "scan_dna"
   params:
     regions: []          # []=full seq; or [[start,end), ...] 0-based
 output:
   dir: "${JOB_DIR}/../runs/rnaseh1_nt_scan"
 plot:
   which: ["position_scatter_and_heatmap","metric_by_mutation_count"]
```

Run a deep mutational scan on codons:
```yaml
job:
  name: rnaseh1_codon_scan
  bio_type: dna
  input:
    refs: "${JOB_DIR}/../inputs/refs.csv"
    name_col: "ref_name"
    seq_col: "sequence"
  permute:
    protocol: "scan_codon"
    params:
      codon_table: "${JOB_DIR}/../inputs/codon_ecoli.csv"
      # region_codons: [0, 100]  # optional [start,end) in codon units
  evaluate:
    metrics:
      - id: llr
        evaluator: evo2_llr
        metric: log_likelihood_ratio
        params:
          model_id: evo2_7b
          device: cuda:0
          precision: bf16
          alphabet: dna
          reduction: mean
  output:
    dir: "${JOB_DIR}/../runs/rnaseh1_codon_scan"
  plot:
    which: ["position_scatter_and_heatmap","metric_by_mutation_count"]
```

Run either one:

```bash
permuter run --job src/dnadesign/permuter/jobs/rnaseh1_nt_scan.yaml   --ref BL21_RNase_H1_wt
permuter run --job src/dnadesign/permuter/jobs/rnaseh1_codon_scan.yaml --ref BL21_RNase_H1_wt
```

---

### Run → Evaluate → Plot → Export

```bash
# Generate variants (one dataset per ref)
permuter run --job src/dnadesign/permuter/jobs/rnaseh1_nt_scan.yaml --ref rnaseh1

# Evaluate (append metrics to the same Parquet)
permuter evaluate --data src/dnadesign/permuter/runs/rnaseh1_nt_scan/rnaseh1/records.parquet \
  --with llr:evo2_llr:log_likelihood_ratio

# Make plots
permuter plot --data src/dnadesign/permuter/runs/rnaseh1_nt_scan/rnaseh1/records.parquet \
  --which position_scatter_and_heatmap metric_by_mutation_count \
  --metric-id llr

# Optional CSV export
permuter export --data src/dnadesign/permuter/runs/rnaseh1_nt_scan/rnaseh1/records.parquet \
  --fmt csv --out src/dnadesign/permuter/runs/rnaseh1_nt_scan/rnaseh1/records.csv

# Validate the dataset
permuter validate --data src/dnadesign/permuter/runs/rnaseh1_nt_scan/rnaseh1/records.parquet --strict
```

**What you get**

```
runs/rnaseh1_nt_scan/rnaseh1/
  records.parquet         # USR core + permuter columns + metric(s)
  REF.fa                  # the reference DNA
  plots/
    position_scatter_and_heatmap.png
    metric_by_mutation_count.png
```

---

### CLI reference

```
permuter run      --job PATH [--ref NAME] [--out DIR]
permuter evaluate --data PATH [--with id:ev[:metric]]... [--job PATH]
permuter plot     --data PATH [--which name]... [--metric-id ID]
permuter export   --data PATH --fmt csv|jsonl --out PATH
permuter validate --data PATH [--strict]
permuter inspect  --data PATH [--head N]
```

* `run` resolves paths relative to the job YAML; `${JOB_DIR}` and env vars are expanded.
* `evaluate` writes `permuter__metric__<id>` columns into the same Parquet.
* `plot` can pick a specific metric with `--metric-id`.

---

### Built‑ins

**Protocols**

* `scan_dna` — single nucleotide substitutions (A↔C/G/T) over regions
* `scan_codon` — codon substitutions (requires a codon table CSV)
* `scan_stem_loop` — hairpin generator (seeded cap, extend/rebuild modes)

**Evaluators**

* `placeholder` — deterministic pseudo‑scores (for smoke tests)
* `evo2_ll` — log likelihood
* `evo2_llr` — log likelihood ratio vs reference

**Plots**

* `position_scatter_and_heatmap`
* `metric_by_mutation_count`

---

### Data model

Each dataset is a single Parquet file with a **USR core** and a **Permuter** namespace:

**USR core columns**

* `id`: `sha1(bio_type|sequence)`
* `bio_type`: `"dna"` or `"protein"`
* `sequence`: the full variant sequence
* `alphabet`: `"dna_4"` or `"protein_20"`
* `length`: integer length of `sequence`
* `source`: provenance string
* `created_at`: UTC timestamp

**Permuter namespace (`permuter__*`)**

* `permuter__job`, `permuter__ref`, `permuter__protocol`
* `permuter__var_id`: deterministic BLAKE2b of (job, ref, protocol, sequence, modifications)
* `permuter__round`: `1` (single-pass in core)
* `permuter__modifications`: list of human-readable tokens (e.g., `["nt pos=12 wt=A alt=T"]`)
* Protocol-specific flat fields (e.g., `permuter__nt_pos`, `permuter__aa_pos`, `permuter__hp_length_paired`, …)
* Metrics appended by `evaluate` live in `permuter__metric__<id>` (floats)

---

### Extending

Add files under `src/dnadesign/permuter/src/…`

* Protocols: subclass `Protocol`, implement `validate_cfg()` + `generate()`
* Evaluators: subclass `Evaluator`, implement `score()`
* Plots: export `plot(elite_df, all_df, output_path, job_name, ref_sequence=None, metric_id=None, evaluators="")`

---

@e-south

