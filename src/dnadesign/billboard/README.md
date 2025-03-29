## *billboard*

**billboard** quantifies the regulatory diversity of dense-array–derived DNA sequences based on transcription factor binding site (TFBS) composition and distribution. It processes batches of sequences, extracts motif information, computes core diversity metrics, and writes a `diversity_summary.csv` file to support downstream workflows.

**billboard** helps to answer: ***how broad, balanced, distinct, and spatially diffuse are the regulatory elements across the dense array library?***

### Pipeline

Given a `.pt` file containing sequence dictionaries (in the sibling `sequences/` directory), **billboard**:

1. Parses TFBS annotations to extract which transcription factors are present in each sequence.
2. Constructs strand-specific positional occupancy maps for each TF.
3. Computes four scalar diversity metrics:
   - **TF Richness** – Compositional breadth.
   - **Inverted Gini Coefficient** – Usage balance across TFs.
   - **Mean Jaccard Dissimilarity** – Combinatorial diversity of TF rosters.
   - **Positional Entropy** – How diffuse each TF's binding is across the sequence.
4. Optionally computes a weighted composite score.
5. Writes results to `diversity_summary.csv`.
6. Optionally produces diagnostic plots.

### Simple Usage

1. **Edit your YAML config** (`configs/example.yaml`):
    ```yaml
    billboard:
      output_dir_prefix: example_library
      pt_files:
        - example_sequences
      save_plots: true
      dry_run: false
      composite_weights:
        tf_richness: 0.4
        1_minus_gini: 0.3
        mean_jaccard: 0.3
        median_tf_entropy: 0.1
    ```

2. **Run the analysis:**
    ```bash
    python billboard/main.py
    ```

    Results will appear under:
    ```
    batch_results/example_library_YYYYMMDD/
    ```

### Core Diversity Metrics

Each metric captures a different aspect of library diversity. All are scalar, interpretable, and suitable for low-N analysis.

#### 1. TF Richness — *Compositional Breadth*

- **Definition**: Counts how many unique TFs are present across the library.
- **Computation**: Let `T = {t₁, t₂, ..., tₖ}` be the union of TFs; then `TF Richness = |T|`.
- **Summary**: Measures how many distinct TFs are represented, regardless of how often or where they appear.

#### 2. Inverted Gini Coefficient — *Usage Balance*

- **Definition**: Quantifies how evenly TFs are used based on frequency.
- **Computation**:  
  `Gini = (∑₁ⁿ ∑₁ⁿ |fᵢ - fⱼ|) / (2n ∑ f)`,  
  then take `1 - Gini` to reward evenness.
  ```python
  # Example
  tf_counts = [10, 10, 10]  # TF usage counts across the library is perfectly even
  n = len(tf_counts)
  total = sum(tf_counts)

  gini_numerator = sum(abs(x - y) for x in tf_counts for y in tf_counts)
  gini = gini_numerator / (2 * n * total)
  inverted_gini = 1 - gini

  print(f"Gini: {gini:.2f}, Inverted Gini: {inverted_gini:.2f}")  # → Gini: 0.00, Inverted Gini: 1.00

  # Now try an imbalanced example:
  tf_counts = [25, 5, 0]
  n = len(tf_counts)
  total = sum(tf_counts)

  gini_numerator = sum(abs(x - y) for x in tf_counts for y in tf_counts)
  gini = gini_numerator / (2 * n * total)
  inverted_gini = 1 - gini

  print(f"Gini: {gini:.2f}, Inverted Gini: {inverted_gini:.2f}")  # → Gini: 0.56, Inverted Gini: 0.44
  ```

- **Summary**: Captures inequality in TF usage—higher values indicate more balanced distribution, but does not consider TF identity or binding location.

#### 3. Mean Jaccard Dissimilarity — *Combinatorial Diversity*

- **Definition**: Measures how distinct TF rosters are between sequences.
- **Computation**:  
  For two sequences A and B with TF sets `T_A` and `T_B`:  
  `D(A, B) = 1 - |T_A ∩ T_B| / |T_A ∪ T_B|`

  Average `D(A, B)` across all sequence pairs.
  ```python
  # Example
  Seq1_TFs = {"CRP", "FadR", "LexA"}
  Seq2_TFs = {"FadR", "ArcA"}

  intersection = Seq1_TFs & Seq2_TFs        # {"FadR"}
  union = Seq1_TFs | Seq2_TFs               # {"CRP", "FadR", "LexA", "ArcA"}

  jaccard = len(intersection) / len(union)  # 1 / 4 = 0.25
  dissimilarity = 1 - jaccard               # 0.75
  ```

- **Summary**: Reflects the diversity of TF combinations across sequences; robust at small sample sizes, but ignores frequency and position.

#### 4. Median Positional Entropy — *Spatial Diffusion*

- **Definition**: Assesses how widely each TF binds across the sequence.
- **Computation**:
  - For each TF, build a position-wise count vector `P = [p₁, p₂, ..., p_L]`, normalize it to `P̂`.
  - Compute entropy: `H = -∑ P̂ᵢ log₂(P̂ᵢ)`, normalized by `log₂(L)`.
  - Average forward and reverse strand entropy; take the **median** across TFs.
  ```python
  # Example
  import numpy as np
  # Two TFs binding across a sequence of length 5
  P_TF1 = np.array([10, 0, 0, 0, 0])     # Clustered binding at position 0
  P_TF2 = np.array([2, 2, 2, 2, 2])       # Uniformly spread binding

  def entropy(P):
      P_hat = P / P.sum()
      return -np.sum(P_hat * np.log2(P_hat))

  H1 = entropy(P_TF1)
  H2 = entropy(P_TF2)
  L = 5
  H1_norm = H1 / np.log2(L)  # → 0.00 (completely focused)
  H2_norm = H2 / np.log2(L)  # → 1.00 (maximally diffuse)

  print(f"TF1 entropy: {H1_norm:.2f}")
  print(f"TF2 entropy: {H2_norm:.2f}")
  # Median Positional Entropy = median([H1_norm, H2_norm]) = 0.50
  ```

- **Summary**: Captures whether TFs bind diffusely or cluster at specific positions. Robust to strand orientation and motif redundancy, though sparse TFs may skew results.

### Composite Metric

To create a single scalar summary of diversity, **billboard** supports a **weighted sum** of the core metrics:

```yaml
# YAML config snippet
composite_weights:
  tf_richness: 0.5
  1_minus_gini: 0.3
  mean_jaccard: 0.2
  median_tf_entropy: 0.1
```
```python
billboard_weighted_sum = w1 * tf_richness + w2 * (1 - gini) + w3 * mean_jaccard + w4 * median_tf_entropy
```

### Output

After running, **billboard** writes a results folder under `batch_results/`:

### Key Deliverable
- `csvs/diversity_summary.csv`
  - `tf_richness`
  - `1_minus_gini`
  - `mean_jaccard`
  - `median_tf_entropy`
  - `billboard_weighted_sum`

### Optional Plots
- `tf_frequency_barplot.png`
- `tf_occupancy_combined.png`
- `motif_length_density.png`

### Module Overview

- `main.py`: Orchestrates loading config, processing sequences, and saving outputs.
- `core.py`: Computes diversity metrics and TF occupancy.
- `plot_helpers.py`: Generates optional plots.
- `summary.py`: Writes CSVs, including the diversity summary.