## billboard

![billboard banner](assets/billboard-banner.svg)

**billboard** quantifies regulatory diversity in dense‑array DNA libraries by extracting TFBS content and computing a small set of scalar “diversity” metrics and diagnostic plots.

> **The pipeline helps to answer:**
> *How broad, balanced, distinct, and spatially diffuse are the transcription‑factor binding sites across a library of sequences?*

### Quick Start

Assuming you have a **densegen**-derived `.pt` file under `sequences/` containing a Python list of sequence dicts:

```bash
python billboard/main.py
```

By default, outputs will be written to
```
batch_results/<output_dir_prefix>_YYYYMMDD/
```

### Pipeline Overview

1. **Load configuration** (`configs/example.yaml`).
2. **Load sequences** from one or more `.pt` files.
3. **Parse TFBS annotations** (`meta_tfbs_parts` / `meta_tfbs_parts_in_array`).
4. **Build motif strings** (ordered comma‑delimited TF+strand lists).
5. **Compute occupancy matrices** for forward/reverse strands.
6. **Compute core diversity metrics**:
   - TF Richness
   - Inverted Gini Coefficient
   - Minimum Jaccard Dissimilarity
   - Minimum Positional Entropy
   - Minimum Motif‑String Levenshtein
7. **Write CSV summaries** (`diversity_summary.csv`, `tf_coverage_summary.csv`, …).
8. **(Optional)** Generate diagnostic plots.

### Configuration

```yaml
# example.yaml
billboard:
  # Prefix for your output folder in billboard's batch_results
  output_dir_prefix: example_library

  # Specify one or more PT entries under sequences/
  pt_files:
    - example_sequences
  save_plots: true
  skip_aligner_call: true
  # Figure DPI for saved PNGs
  dpi: 300

  # Which metrics to compute (order doesn’t matter)
  diversity_metrics:
    - tf_richness
    - 1_minus_gini
    - min_jaccard_dissimilarity
    - min_tf_entropy
    - min_motif_string_levenshtein

  # Penalties for motif‑string Levenshtein
  motif_string_levenshtein:
    tf_penalty: 1.0
    strand_penalty: 0.5
    partial_penalty: 0.8
```

### Core Diversity Metrics

Each metric is a **single scalar** summarizing one aspect of library diversity. **Higher values correspond to greater diversity.**

---

#### **Richness** — *TF Compositional Breadth*

Count of **unique** TFs placed across all sequences:

  TF_Richness = | ⋃ₛ Tₛ |

  where Tₛ is the set of TFs in sequence s.

> A larger TF Richness means more distinct transcription factors appear anywhere in your library—i.e., broader regulatory set.

---

#### **Inverted Gini Coefficient** — *TF Usage Balance*

Measures how evenly TFs occur across all sequences.

1. Compute raw Gini:

    G = (∑_{i=1}ⁿ ∑_{j=1}ⁿ |fᵢ - fⱼ|) / (2 * n * ∑_{k=1}ⁿ fₖ)


2. Invert to reward evenness:

    1 - G

    where fᵢ is the total count of TF i across all sequences.

> Values near 1 indicate TFs are used with similar frequency (balanced usage), whereas values near 0 indicate dominance by a few TFs (unequal usage).

---

#### **Min Jaccard Dissimilarity** — *TF Combinatorial Diversity*

Find the **smallest** pairwise dissimilarity between any two sequences’ TF sets:

  Dᵢⱼ = 1 - |Tᵢ ∩ Tⱼ| / |Tᵢ ∪ Tⱼ|

  min_over_i<j (Dᵢⱼ)


where Tᵢ and Tⱼ are the TF sets of sequences i and j.

> A higher minimum dissimilarity means **every** pair of sequences differs substantially in TF composition; a zero means at least one pair is identical.

---

#### **Min Positional Entropy** — *TF Spatial Diffusion*
Assess how “focused” the most localized TF is along the sequence.

1. **Counts → Probabilities**
   - Let pₖ be the binding count of a given TF at position k (forward or reverse).

    Normalize:

    p̂ₖ = pₖ / (∑_{i=1}ᴸ pᵢ)    with   ∑ₖ p̂ₖ = 1

2. **Entropy & Normalization**
    H = - ∑ₖ p̂ₖ · log₂(p̂ₖ)

    H_norm = H / log₂(L)

3. **Combine & Summarize**
  - Compute H_norm for forward and reverse strands, average per TF.
  - Take the minimum across TFs.

> `H_norm = 0` if a TF binds only at one spot (highly focused), and `1` if it binds uniformly. By taking the minimum, a low score flags that at least one TF is very localized; a high score means **every** TF is spatially diffuse.

---

#### **Min Motif‑String Levenshtein** — *Sequence Architectural Diversity*

1. Represent each sequence as a strand-aware, ordered token list, e.g.,

    ["crp+", "gadX-", ...].

2. Compute pairwise, length‑normalized Levenshtein distance:

    d_norm(sᵢ, sⱼ) = Levenshtein(sᵢ, sⱼ) / max(|sᵢ|, |sⱼ|)

3. Report min_over_i<j (Dᵢⱼ) over all pairs.

> A higher minimum distance indicates that even the two most similar sequences differ in motif architecture, whereas a value of zero means at least one pair shares an identical ordering of motifs derived from the same transcription factors.

---

#### **Min Needleman–Wunsch Dissimilarity** — *Global Sequence Diversity*

Evaluates global sequence‐level diversity via optimal alignment.

1.	Compute normalized global alignment similarity for each pair (i,j):

    Sᵢⱼ = NW(sᵢ, sⱼ) / max(ℓᵢ, ℓⱼ)

    where NW(sᵢ, sⱼ) is the Needleman–Wunsch alignment score and ℓᵢ is the length of sequence sᵢ.

2.	Convert to dissimilarity:

    Dᵢⱼ = 1 - Sᵢⱼ

3.	Report the minimum (D_{ij}) over all (i<j).

> A high minimum NW dissimilarity means even the two most similar sequences are quite different at the nucleotide level, whereas a zero indicates at least one pair is perfectly alignable.

---

### Output Structure

After running, you’ll find under `batch_results/<prefix>_YYYYMMDD/`:

```
csvs/
  ├ diversity_summary.csv     # core metrics per library
  ├ tf_coverage_summary.csv   # TF frequencies
  └ summary_metrics.csv       # basic counts (sequences, TFBS instances)

plots/  (if save_plots: true)
  ├ tf_frequency.png
  ├ occupancy.png
  ├ motif_length.png
  ├ tf_entropy_hist.png
  ├ gini_lorenz.png
  └ jaccard_hist.png
```

### Module Overview

- **`main.py`**
  Manages config loading, sequence processing, metric computation, and I/O.

- **`core.py`**
  Implements:
  - Sequence loading & validation
  - TFBS parsing & motif‑string building
  - Occupancy matrix assembly
  - Diversity metrics (`compute_core_metrics`)

- **`summary.py`**
  Writes CSV summaries:
  - `diversity_summary.csv`
  - `tf_coverage_summary.csv`
  - `summary_metrics.csv`

- **`plot_helpers.py`**
  Optional diagnostic plots:
  - TF frequency barplot
  - Occupancy heatmaps
  - Motif‑length density
  - Positional occupancy histograms
  - Lorenz (Gini) curves
  - Jaccard dissimilarity histogram

- **`by_cluster.py`**
  Computes metrics per Leiden cluster of sequences, producing a characterization scatter plot. This modules expects an input `.pt` file which has already been passed through the **clustering** pipeline, where each entry receives a cluster ID.
