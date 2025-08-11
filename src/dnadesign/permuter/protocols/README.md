### Permuter Protocols

*A compact guide to configuring and using permutation **protocols** in `dnadesign.permuter`.*
This document covers the common contract, plus succinct how-tos for:

* `scan_dna` (nucleotide saturation)
* `scan_codon` (codon swap by amino-acid)
* `scan_stem_loop` (hairpin growth/rebuild)

---

#### Protocols in one picture

* Every protocol implements a common **ABC** with two methods:

  * `validate_cfg(params)` — assertive, protocol-specific validation
  * `generate(ref_entry, params, rng)` — **streaming** iterator of variant dicts
* Runner and selectors are protocol-agnostic. Protocols emit:

  * `sequence` (full, uppercase A/C/G/T)
  * `modifications` (list\[str]; first entry is a compact, human-readable summary)
  * Flat, namespaced metadata keys (`nt_*`, `codon_*`, `hp_*`, `gen_*`)

---

#### Minimal job skeleton

```yaml
permuter:
  experiment: { name: demo }

  jobs:
    - name: my_job
      input_file: dnadesign/permuter/input/refs.csv
      references: ["my_ref"]

      run: { mode: full }    # full | analysis | auto

      permute:
        protocol: scan_dna   # or scan_codon | scan_stem_loop
        params: { ... }      # protocol-specific (see below)

      evaluate:
        metrics:
          - id: ll
            name: log_likelihood
            evaluator: placeholder
            goal: max
            norm: { method: rank, scope: round }

      select:
        objective: { type: weighted_sum, weights: { ll: 1 } }
        strategy:  { type: top_k, k: 10, include_ties: true }
```

---

#### `scan_dna` — nucleotide saturation

**What it does**:
For each position in the chosen region(s), substitute the nucleotide with the other three bases and emit variants.

**Params**

```yaml
permute:
  protocol: scan_dna
  params:
    # Choose one:
    region: [START, END]      # 0-based, END exclusive; omit → full sequence
    # or multiple regions
    regions:
      - [START1, END1]
      - [START2, END2]
```

**Behavior**

* Validates input contains only A/C/G/T (uppercase normalized).
* For each position `i` in region(s), emits up to 3 variants (swap to the other bases).
* **Variant metadata**

  * `nt_pos` (int, 0-based index)
  * `nt_wt`, `nt_alt` (single chars)
* **Modifications summary** (first item):

  * `nt pos={nt_pos} wt={nt_wt} alt={nt_alt}`

**Example**

```yaml
permute:
  protocol: scan_dna
  params:
    region: [100, 120]
```

---

#### `scan_codon` — codon-level substitutions

**What it does**:
At each codon in the chosen region, replace with the **most frequent codon of every other amino acid** using a lookup CSV.

**Params**

```yaml
permute:
  protocol: scan_codon
  params:
    codon_table: path/to/codon_usage.csv   # columns: codon, amino_acid, frequency
    region_codons: [START, END]            # in codon units; omit → full CDS
```

**Behavior**

* Requires sequence length divisible by 3 in the region.
* For codon index `ci`, finds the WT amino acid; for every *other* amino acid, picks its **top codon** and emits a variant.
* **Variant metadata**

  * `codon_index` (0-based)
  * `codon_wt`, `codon_new` (e.g., `ATG`, `GAA`)
  * `codon_aa` (target amino acid symbol/string)
* **Modifications summary**:

  * `codon i={codon_index} wt={codon_wt} new={codon_new} aa={codon_aa}`

**Example**

```yaml
permute:
  protocol: scan_codon
  params:
    codon_table: dnadesign/permuter/input/ecoli_codon_usage.csv
    region_codons: [0, 50]
```

---

#### `scan_stem_loop` — hairpin growth/rebuild

**What it does**:
Inserts or replaces a **stem–loop (hairpin)** in a specified region and **systematically varies stem length**, generating multiple **unique** replicates per length. Growth can be anchored at the **cap** (loop) or **base**. Base sampling is **GC-aware**; **mismatches** and **rare indels** may be introduced during growth.

**Key ideas**

* `region` is either an **insert index** or a **\[start, end]** replacement window.
* `seed` includes explicit **upstream stem**, **cap** (loop; length ≥ 3), and **downstream stem**.
* `program` sets **mode** (`extend`/`rebuild`), **length schedule** (inclusive stop), **anchor** point, **replicates per length**, and rates for **GC**, **mismatch**, **indel**.
* Dedupe is **hairpin-only**: uniqueness is determined by the assembled `upstream + cap + downstream`.

**Params**

```yaml
permute:
  protocol: scan_stem_loop
  params:
    # insert at i  OR  replace [start, end]
    region: [START, END] | INSERT_INDEX

    seed:
      upstream_stem: "ACGT..."    # 5'→3' as appears it will appear in full sequence (may be empty)
      cap: "GAAAC"                # REQUIRED, len ≥ 3
      downstream_stem: null       # null ⇒ reverse complement of upstream_stem

    program:
      mode: extend | rebuild
      stem_len: { start: 6, stop: 20, step: 1 }  # inclusive stop, paired columns
      anchor: cap | base
      samples_per_length: 3
      gc_target: 0.55
      mismatch: { rate: 0.05 }
      indel:    { rate: 0.01 }

    dedupe:
      scope: hairpin
      retry_per_length: 32

    rng_seed: 123                 # optional
```

**Variant metadata (flat)**

* Region & mode: `hp_region_start`, `hp_region_end`, `hp_mode`, `hp_anchor`
* Length & composition: `hp_length_paired`, `hp_cap_len`, `hp_up_len`, `hp_down_len`, `hp_asymmetry`
* Quality metrics: `hp_mismatch_frac`, `hp_gc_frac_paired`, `hp_longest_match_run`
* Parts & identity: `hp_upstream`, `hp_cap`, `hp_downstream`, `hp_subseq`, `hp_hash`
* Generation knobs: `gen_replicate_idx`, `gen_gc_target`, `gen_mismatch_rate`, `gen_indel_rate`
* **Modifications summary**:

  * `hp L={hp_length_paired} anchor={hp_anchor} mode={hp_mode} region={hp_region_start}:{hp_region_end} cap={hp_cap_len} asym={hp_asymmetry} mis={hp_mismatch_frac:.3f} gc={hp_gc_frac_paired:.3f} rep={gen_replicate_idx} hash={hp_hash}`

**Quick examples**

1. **Extend a natural hairpin inside a window**

```yaml
permute:
  protocol: scan_stem_loop
  params:
    region: [120, 180]
    seed:
      upstream_stem: "ACGTAC"
      cap: "GAAAC"
      downstream_stem: null   # auto-RC
    program:
      mode: extend
      stem_len: { start: 6, stop: 20, step: 1 }
      anchor: cap
      samples_per_length: 3
      gc_target: 0.55
      mismatch: { rate: 0.05 }
      indel:    { rate: 0.01 }
    dedupe: { scope: hairpin, retry_per_length: 32 }
```

2. **Rebuild from scratch at an insertion point**

```yaml
permute:
  protocol: scan_stem_loop
  params:
    region: 250   # insert here
    seed:
      upstream_stem: ""       # start empty
      cap: "GAAAC"
      downstream_stem: ""     # or null to mirror upstream as it grows? (for rebuild keep explicit)
    program:
      mode: rebuild
      stem_len: { start: 0, stop: 16, step: 1 }
      anchor: base
      samples_per_length: 4
      gc_target: 0.50
      mismatch: { rate: 0.02 }
      indel:    { rate: 0.005 }
    dedupe: { scope: hairpin, retry_per_length: 32 }
```

---

#### Determinism, dedupe, and IDs

* **Deterministic RNG:** runner injects a derived seed; protocol derives per-length/per-replicate seeds so identical configs → identical sequences.
* **Dedupe (hairpin-only):** uniqueness is by `hp_hash = H(upstream + cap + downstream)`; per-length replicate quotas are filled without duplicates when possible.
* **`modifications` first entry:** compact, key=value summary string (protocol-specific).
* **`var_id`:** is computed upstream (hash of job/ref/protocol/sequence/modifications).

---

#### Common validation & errors (examples)

* `region must be an integer or [start, end]`
* `region out of bounds for sequence length {N}: {region}`
* `scan_codon: region must align to codon boundaries`
* `scan_codon: missing or invalid codon_table`
* `scan_stem_loop: seed.cap is required and must be A/C/G/T uppercase`
* `scan_stem_loop: seed.cap length must be ≥ 3`
* `scan_stem_loop: program.stem_len.start must be ≥ seed paired length (extend mode)`

---

#### Best practices

* Keep sequences **uppercase A/C/G/T**.
* Prefer **one protocol per job** for clean outputs and reproducibility.
* For `scan_stem_loop`, start with **modest** `samples_per_length` (e.g., 3–5) and increase only if evaluator variance warrants it.
* Check WARN logs for dedupe saturation at certain lengths.

