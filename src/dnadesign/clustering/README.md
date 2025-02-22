## clustering

**clustering** utilizes [Scanpy](https://scanpy.readthedocs.io/en/stable/) for cluster analysis on nucleotide sequences stored in the sibling **sequences** directory as `.pt` files (assumed to be lists of dictionaries). By default, it uses the mean-pooled output logits of [Evo 2](https://github.com/ArcInstitute/evo2) along the sequence dimension as input. The pipeline performs high-dimensional feature extraction, generates UMAP embeddings, and applies Leiden clustering. The pipeline also supports downstream analyses, including cluster composition, diversity assessment, and differential feature analysis, while offering flexible plotting options.

### Modules

- **main.py**  
  The entry point of the pipeline. It:
  - Loads the YAML configuration.
  - Reads data entries from **sequences**.
  - Extracts features (e.g., defaulted to `evo2_logits_mean_pooled`) and creates an AnnData object.
  - Computes UMAP embeddings using Scanpy and performs Leiden clustering.
  - Saves selected clusters along with summary YAML files.
  - Calls analysis modules and saves resulting plots and CSVs into `clustering/batch_results/<batch_name>`.

- **plotter.py**  
  Handles UMAP plotting:  
  - Supports multiple hue methods, including `"input_source"`, `"numeric"`, and `"type"`, among others. The corresponding keys must be present in each sequence entry's dictionary.  
  - For continuous hues, applies a colormap (e.g., 'viridis') and adds a colorbar.  
  - Supports plotting multiple datasets with distinct numeric hues (e.g., different promoter MPRA datasets with transcription strength labels). When combining datasets, numeric labels are standardized using **robust standardization** before plotting:  

    $$
    X' = \frac{X - \text{median}(X)}{\text{IQR}(X)}
    $$

- **cluster_select.py**  
  Selects clusters for export based on configuration options (e.g., number of clusters, order, or custom selections).

- **cluster_composition.py, diversity_analysis.py, differential_feature_analysis.py**

  These modules conduct downstream analyses:
  - **Cluster Composition**: Creates a contingency table and stacked bar plot to show the proportion of upstream groups per cluster.
  - **Diversity Assessment**: Computes diversity metrics (Shannon entropy, Simpson’s index) and generates grouped bar plots.
  - **Differential Feature Analysis**: Identifies over-represented markers per cluster and exports the results as CSV.

### Configuration

The pipeline is configured via a YAML file (e.g., `configs/example.yaml`). Key configuration elements include:

- **batch_name**: A unique identifier for the run.
- **input_sources**: List of directories containing `.pt` files. Each `.pt` file should be a list of dictionaries. Each dictionary must contain keys such as `"id"`, `"sequence"`, and various metadata fields.
- **umap**: Settings for UMAP generation, including plot dimensions, transparency (alpha), and hue options. For example, hue can be set to:
  - `"input_source"`: Uses the source directory name.
  - `"numeric"`: Extracts a numeric value (with normalization options such as `"robust"`).
  - `"type"`: Extracts categorical hue from `"meta_part_type"`.
- **analysis**: Toggles whether to perform cluster composition, diversity, and differential feature analyses.


### Usage

1. **Run the Pipeline:**  
   From the **clustering** project root, execute:
   ```bash
   python main.py --config configs/example.yaml
   ```
   Use the `--dry_run` flag to preview execution without saving outputs.

4. **Output:**  
    Results, including clusters, plots, and CSV files, are saved in `clustering/batch_results/<batch_name>_<hue_method>/`. Additionally, any exported clusters are saved as `.pt` files with a summary YAML in the **sequences** directory.
