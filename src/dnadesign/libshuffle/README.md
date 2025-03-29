# LibShuffle – Developer Specification Document

## 1. Overview

LibShuffle is a module that iteratively subsamples sequence libraries (stored in a single .pt file) and computes diversity metrics. It is a sibling to BuildBoard but has been refactored to use a new set of core diversity metrics derived from a file named `diversity_summary.csv`. These metrics are then combined into a composite score via either a PCA or weighted sum strategy. The composite score (which replaces the previously computed raw “Billboard” metric) is used for thresholding and is displayed on the scatter plot (y-axis) and stored in the output details.

---

## 2. Key Goals & Changes

- **New Core Diversity Metrics:**  
  Replace the old entropy metrics with values read from `diversity_summary.csv`. The agreed core metrics are:
  - **tf_richness**  
    *Type:* Scalar  
    *Definition:* Number of unique TFs in the library.
  - **1_minus_gini**  
    *Type:* Scalar  
    *Definition:* Inverted Gini coefficient for TF frequency distribution.
  - **mean_jaccard**  
    *Type:* Scalar  
    *Definition:* Mean pairwise Jaccard dissimilarity of TF rosters.
  - **median_tf_entropy**  
    *Type:* Scalar  
    *Definition:* Aggregated (median) positional entropy across TFs.
  - **billboard_weighted_sum** (optional)  
    *Type:* Scalar (optional composite)  
    *Note:* This is a previously generated composite and is not mandatory for the new strategy.

- **Composite Score Strategy:**  
  The new pipeline allows users to compute a composite diversity score from the selected core metrics. This composite replaces the raw billboard metric:
  - **Normalization:** Must be computed _locally per run_ (using only metrics from the current batch).
  - **Composite Methods:**  
    - **PCA-Based (e.g., “zscore_pca”)**  
      Normalizes selected core metrics (using either z-score or min-max) and computes a principal component.
    - **Weighted Sum (e.g., “minmax_weighted”)**  
      Applies the specified normalization and combines metrics using user-defined weights.
  - **Usage:**  
    - The composite score becomes the y-axis in the scatter plot.
    - It is recorded in the sublibrary (results) YAML and is used in threshold-based joint selection with the Evo2 metric.

- **Configuration Flexibility:**  
  Users can choose between using a composite score (from multiple core metrics) or a single raw metric. The configuration must expose:
  - Whether composite scoring is enabled.
  - The method (e.g., “zscore_pca” or “minmax_weighted”).
  - The normalization strategy (“zscore”, “minmax”, or none).
  - The list of core metrics to include.
  - Weights for each metric (if using weighted sum).

- **Output Handling:**  
  - **Global Summary:** Renamed to `global_summary.yaml`, containing run-wide config parameters and summary data.
  - **Sublibrary Details:** Renamed to `sublibraries.yaml` with per-subsample metrics (including composite score, Evo2 metric, selected indices, and—if applicable—PCA artifacts).
  - **Visualization:** A scatter plot (Evo2 on x-axis, composite (billboard) on y-axis) is generated.
  - **Selected Subsamples:** Previously saved PT files for isolated sub-libraries are now stored within the batch results directory.

---

## 3. System Architecture & Modules

The LibShuffle module is structured as follows:

```
dnadesign/
├── README.md
└── src/
    └── dnadesign/
        ├── configs/
        │   └── example.yaml         # Global configuration file (includes libshuffle & billboard settings)
        ├── libshuffle/
        │   ├── __init__.py
        │   ├── config.py            # Loads and validates YAML configuration
        │   ├── metrics.py           # Computes diversity metrics and composite score
        │   ├── subsampler.py        # Performs iterative subsampling, caching, deduplication, and progress tracking
        │   ├── visualization.py     # Generates scatter plots using computed metrics
        │   ├── output.py            # Writes YAML summaries, sublibrary details, and (optionally) selected .pt files
        │   └── main.py              # CLI entry point for executing the libshuffle pipeline
        ├── sequences/               # Contains input .pt file and may hold output PT files if selected subsamples are saved
        └── ...                      # Other pipelines (densegen, evoinference, etc.)
```

### Module Details

- **config.py:**  
  - Loads the configuration from YAML.
  - Validates that the configuration contains a `libshuffle` key.
  - Provides helper methods (e.g., `get(key, default)`) to access configuration values.

- **metrics.py:**  
  - Reads the `diversity_summary.csv` from temporary Billboard runs.
  - Computes individual core metrics from the CSV.
  - **Composite Score Calculation:**  
    - If `composite_score: true` in the configuration, then:
      - Extracts the selected core metrics.
      - Applies the specified normalization (zscore or minmax) _locally_ on the current run’s data.
      - Combines the normalized values via PCA (if method is “zscore_pca”) or by a weighted sum (if method is “minmax_weighted”).
      - Returns the final composite score as the new billboard metric.
    - If `composite_score: false`, then a single metric (e.g., “1_minus_gini”) is used directly.

- **subsampler.py:**  
  - Validates that each sequence contains a unique `"id"` field.
  - Iteratively draws subsamples (with or without replacement).
  - Uses caching (with a frozenset of sequence IDs) to avoid duplicate subsamples.
  - Computes both the composite billboard metric (via the updated metrics module) and the Evo2 metric (mean pairwise L2 distance).
  - Enforces a maximum number of deduplication attempts.

- **visualization.py:**  
  - Generates a scatter plot:
    - **X-axis:** Evo2 metric.
    - **Y-axis:** Composite billboard metric.
  - Highlights subsamples that pass joint selection thresholds defined in the configuration.
  - Uses Seaborn’s “ticks” style and saves the plot as a PNG with user-specified DPI and filename.

- **output.py:**  
  - Writes two YAML files:
    - **Global Summary (`global_summary.yaml`):** Contains overall run metadata (run time, duration, configuration parameters, number of sequences, etc.).
    - **Sublibraries (`sublibraries.yaml`):** Contains per-subsample details (indices, computed metrics, whether thresholds were passed, and if composite was computed, PCA artifacts such as explained variance and component loadings).
  - Optionally saves selected subsample .pt files in the batch results directory (if enabled in the config).

- **main.py:**  
  - CLI entry point that ties all modules together:
    - Loads the configuration.
    - Resolves the input directory (ensuring exactly one .pt file is present).
    - Loads sequences (validating required fields).
    - Creates an output directory (timestamped under batch_results).
    - Executes subsampling and metrics computation.
    - Generates visualization.
    - Saves output YAMLs and selected subsample files.
    - Logs progress and errors, exiting with a failure code if necessary.

---

## 4. Configuration Specification

Below is the proposed YAML structure for libshuffle. Key sections include the new composite strategy and adjustments to output handling:

```yaml
libshuffle:
  # Input/Output
  input_pt_path: sequences/<input_directory>   # Directory containing exactly one .pt file
  output_dir_prefix: shuffle_sigma70_test        # Used for naming the batch results directory
  save_selected_pt: true                         # If true, save selected subsample .pt files in batch results

  # Subsampling parameters
  subsample_size: 100                            # Default subsample size
  num_draws: 200                                 # Number of subsampling iterations
  with_replacement: false
  random_seed: 42
  max_attempts_per_draw: 10

  # Evo2 Metric Configuration
  evo2_metric:
    type: "log1p_l2"                            # Options: "l2", "log1p_l2", etc.

  # Billboard / Composite Score Configuration
  billboard_metric:
    # Use composite score or a single raw metric
    composite_score: true                        # true: compute composite; false: use a single raw metric
    # When composite_score is true, specify the method and details:
    method: "zscore_pca"                         # Options: "zscore_pca", "minmax_weighted"
    normalize: "zscore"                          # Options: "zscore", "minmax", or null (if no normalization)
    core_metrics:                                # List of core metrics to include (must match columns in diversity_summary.csv)
      - tf_richness
      - 1_minus_gini
      - mean_jaccard
      - median_tf_entropy
    # If using weighted sum, provide weights (ignored for PCA method)
    weights:
      tf_richness: 0.25
      1_minus_gini: 0.25
      mean_jaccard: 0.25
      median_tf_entropy: 0.25

  # Joint selection for thresholding subsamples
  joint_selection:
    enable: true
    method: "threshold"                          # Currently, threshold-based selection
    billboard_min: 0.68                          # Minimum composite billboard metric required
    evo2_min: 0.5                                # Minimum evo2 metric required
    normalize: false                             # (Legacy flag; ensure composite score is used as billboard metric)
    save_top_k: 3                                # Number of top subsamples to save based on combined metrics

  # Plotting configuration
  plot:
    base_color: "gray"                           # Color for regular subsamples
    highlight_color: "red"                       # Color for threshold-passing (selected) subsamples
    alpha: 0.5
    dpi: 600
    filename: scatter_summary.png
```

**Notes:**

- If `composite_score` is set to `false`, then the expectation is that `core_metrics` will contain exactly one metric (e.g., `"1_minus_gini"`) and its raw value will be used as the billboard metric.
- When using the PCA method, transient PCA artifacts (explained variance ratio, component loadings) should be recorded within the per-subsample YAML under a key such as `pca_model_info`.
- Global run configuration (e.g., input file details, total sequences, run time) will be stored in `global_summary.yaml`.

---

## 5. Data Handling Details

- **Input File:**  
  - A single `.pt` file is expected within the user-specified directory.
  - The file must contain a list of dictionaries representing sequences.
  - Each sequence entry must include a unique `"id"` field.
  - For Evo2 metric calculation, each entry must have the `"evo2_logits_mean_pooled"` field.

- **Temporary Data:**  
  - For computing the diversity metrics (via Billboard), a temporary directory is used. The temporary `.pt` file and intermediate outputs (including `diversity_summary.csv`) are written there and discarded after metric extraction.

- **Output Files:**  
  - **Global Summary:** `global_summary.yaml` (contains run-wide parameters and summary information).
  - **Sublibraries Details:** `sublibraries.yaml` (contains per-subsample details, computed metrics, and—if applicable—PCA artifacts).
  - **Visualization:** A scatter plot image file (PNG) is stored in the output directory.
  - **(Optional) Selected Subsamples:** Saved as `.pt` files in the designated output (batch results) directory if joint selection is enabled.

- **Caching:**  
  - Subsamples are cached using a frozenset of sequence IDs to avoid duplicate metric computation. If a duplicate is detected, the subsample is resampled (up to the configured maximum attempts).

---

## 6. Error Handling & Logging

- **Input Validation:**
  - Ensure the input directory exists and contains exactly one `.pt` file.
  - Validate each sequence entry has an `"id"`. If missing, raise a `ValueError`.
  - Validate that each sequence entry contains the `"evo2_logits_mean_pooled"` field for Evo2 computation.

- **Subsampling Deduplication:**
  - If a duplicate subsample is detected, reattempt until the maximum number of attempts (`max_attempts_per_draw`) is reached. If exceeded, log the error and raise a `RuntimeError`.

- **Billboard Processing Errors:**
  - Wrap calls to the Billboard pipeline (used to generate `diversity_summary.csv`) in try/except blocks. Log exceptions and exit if a critical step fails.

- **General Exception Handling:**
  - Use Python’s built-in `logging` module to capture info, warnings, and errors.
  - In the main execution block, catch all exceptions, log the full stack trace, and exit with a non-zero exit code.
  - Progress of subsampling is tracked with `tqdm`, and errors during subsampling should halt further execution with a clear error message.

---

## 7. Testing Plan

A comprehensive testing plan should include:

### Unit Tests
- **Configuration Loader:**  
  - Test that valid YAML files load correctly.
  - Verify that missing required keys (e.g., `libshuffle` or `billboard_metric`) raise appropriate errors.

- **Metrics Module:**  
  - Unit test normalization functions (zscore and min-max) on sample data.
  - Verify composite score calculation for both PCA and weighted sum methods.
  - Test that the correct core metrics are extracted from a sample `diversity_summary.csv` (simulate the CSV with known values).

- **Subsampler:**  
  - Test input validation (ensure every sequence has an `"id"`).
  - Validate that duplicate subsamples are correctly detected and that the maximum attempts limit is enforced.
  - Ensure that computed metrics (composite billboard and Evo2) match expected outputs on sample input data.

- **Visualization:**  
  - Verify that the scatter plot is generated correctly.
  - Test that selected subsamples (those meeting threshold criteria) are highlighted.

- **Output Module:**  
  - Confirm that YAML files (`global_summary.yaml` and `sublibraries.yaml`) are written in the correct format and include all required fields.
  - Test optional saving of selected subsample `.pt` files.

### Integration Tests
- **End-to-End Run:**
  - Create a mock `.pt` file with test sequences (ensuring proper fields are set).
  - Run the entire pipeline from `main.py` and validate:
    - The output directory is created.
    - Global summary and sublibrary YAMLs are generated and contain correct composite and Evo2 metrics.
    - The scatter plot is generated.
    - Joint selection thresholding works as expected.
    - Logging outputs contain correct progress messages.

- **Error Conditions:**
  - Test behavior when:
    - The input directory contains no or multiple `.pt` files.
    - A sequence entry is missing the `"id"` or `"evo2_logits_mean_pooled"` field.
    - Maximum subsampling attempts are exceeded.

### Regression Tests
- Ensure that when `composite_score` is set to `false`, the pipeline correctly uses a single raw metric.

---

## 8. Developer Implementation Checklist

- [ ] **Update Configuration:**  
  - Add new `billboard_metric` and `composite_strategy` sections.
  - Document allowed methods (`zscore_pca` and `minmax_weighted`) and normalization options.

- [ ] **Refactor Metrics Computation:**  
  - Modify `compute_billboard_metric` in `metrics.py` to extract values from `diversity_summary.csv` and then compute the composite score if enabled.
  - Implement local normalization (min-max and zscore) over the current run’s subsamples.
  - Integrate PCA (using a lightweight PCA implementation) to compute the composite score and record PCA artifacts when applicable.

- [ ] **Update Subsampling Logic:**  
  - Ensure deduplication caching works as intended.
  - Replace raw billboard metric usage with the composite score when enabled.

- [ ] **Revise Visualization:**  
  - Update the scatter plot so that the y-axis reflects the composite billboard metric.
  - Ensure joint selection thresholding is based on the new composite value.

- [ ] **Output Adjustments:**  
  - Rename output files to `global_summary.yaml` and `sublibraries.yaml`.
  - Include PCA artifacts under a key such as `pca_model_info` in the sublibrary details if composite scoring is enabled.
  - Remove previous logic that saved the .pt file per isolated sub-library; instead, save them in the batch results directory if enabled.

- [ ] **Implement Comprehensive Logging & Error Handling:**  
  - Use the logging module and tqdm as specified.
  - Wrap critical operations in try/except blocks.

- [ ] **Testing:**  
  - Write unit tests for each module.
  - Create integration tests using mock .pt input data.
  - Validate error conditions and edge cases.

- [ ] **Documentation & README Updates:**  
  - Update the README to reflect new configuration options, composite strategies, and expected file structure.

---

## 9. Future Considerations

- **Extensibility:**  
  - The design should allow additional composite strategies to be added easily.
  - Further metrics can be incorporated in future iterations by updating the `core_metrics` list and corresponding computation logic.

- **Performance:**  
  - Monitor the performance of the iterative subsampling process.  
  - Consider parallelization if the number of draws is large.

- **Model Artifacts:**  
  - Although PCA artifacts are small, include them in the sublibrary YAML for debugging and transparency.

