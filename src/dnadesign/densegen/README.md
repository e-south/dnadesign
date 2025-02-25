# densegen – Dense Array Generator

**densegen** is a DNA sequence design pipeline built for batch assembly of synthetic bacterial promoters featuring densely packed transcription factor binding sites. It wraps the [**dense-arrays**](https://github.com/e-south/dense-arrays) integer linear programming package and leverages curated transcription factor (TF) to binding site (TFBS) mappings provided by the [**deg2tfbs**](https://github.com/e-south/deg2tfbs) repository.

**densegen** automates the process of:
1. **Data Ingestion:**
  
    Reads `tf2tfbs_mapping.csv` files from deg2tfbs (either from **pipeline/tfbsfetcher** or **analysis** subdirectories) or directly from PyTorch `.pt` files.

2.  **Sampling:**
  
    Randomly samples TFs and their corresponding binding sites.

3. **Optimization:** 

    Generates synthetic promoter sequences using the **dense-arrays** package.

4. **Output:** 
  
    Saves dense arrays along with their metadata as PyTorch `.pt` files in a structured batch directory, while tracking progress via YAML status files.

### Directory Layout

```plaintext
dnadesign/
├── __init__.py               
├── config/
│   └── example.yaml           # User-defined configurations.
├── utils.py                   # Central utilities (paths, constants, etc.).
├── densegen/
│   ├── __init__.py            
│   ├── main.py                # CLI entry point.
│   ├── data_ingestor.py       # Ingests CSV or PT data.
│   ├── sampler.py             # Samples TFs and selects binding sites.
│   ├── optimizer_wrapper.py   # Wraps the dense-arrays Optimizer.
│   ├── progress_tracker.py    # Tracks batch sequence generation in YAML.
│   └── batches/
│       └── seqbatch_<id>.pt   # Timestamped directories storing outputs.
└── sequences/                 # Final sequence outputs.
```

### Simple Usage

1. **Clone the deg2tfbs Repository:**  
   Obtain the curated `tf2tfbs_mapping.csv` files by cloning [deg2tfbs](https://github.com/e-south/deg2tfbs) and placing it as a sibling directory to **dnadesign**.

2. **Configure Parameters:**  
   Update your custom YAML configuration (e.g., `mycustomparams.yaml`) with desired input/output paths, batch IDs, solver preferences, sequence length, and design options. For example:

    ```yaml
    # densegen/configs/mycustomparams.yaml
    densegen:
      input_sources:
        - type: "deg2tfbs_pipeline_tfbsfetcher" # Data source directories (from deg2tfbs).
          name: "all_TFs"
          path: "tfbsbatch_20250223_All"
      output_dir: "sequences"                   # Output directory for generated sequences.           
      progress_file: "progress_status.yaml"     # Progress tracking file.  
      solver: "GUROBI"                          # Preferred solver; falls back to CBC if necessary.  
      solver_options:
        - "Threads=16"
        - "TimeLimit=5"
      sequence_length: 120                      # Length of each dense array.                         
      quota: 10000                              # Number of arrays to generate per batch.        
      subsample_size: 16
      round_robin: true                         # Enable round-robin interleaving among sub-batches.
      arrays_generated_before_resample: 1       # Generate multiple arrays per TF–TFBS sample.       
      fixed_elements:                           # Positional constraints for promoter design.
        promoter_constraints:
          - name: "sigma70_consensus"
            upstream: "TTGACA"
            downstream: "TATAAT"
            spacer_length: [16, 18]
            upstream_pos: [10, 100]
          - name: "sigma32_consensus"
            upstream: "TGTCGCCCTTGAA"
            downstream: "CCCCATTTA"
            spacer_length: [14, 16]
            upstream_pos: [10, 100]
        side_biases:
          left: []
          right: []
      unique_tf_only: true                       # One binding site per TF per sequence.
      fill_gap: true                             # Pad sequences to the desired length.
      fill_gap_end: "5prime"     
      fill_gc_min: 0.40
      fill_gc_max: 0.60
    ```

3. **Run densegen:**  
   Execute the main module via the command line:
   ```bash
   cd densegen
   python main.py
   ```

   For each defined `tf2tfbs_mapping.csv` source, **densegen** will:
   - Load and the mapping data.
   - Sample binding sites and prepare the motif library.
   - Generate promoter sequences using the ILP-based optimizer.
   - Track progress and save outputs as PyTorch `.pt` files in designated batch directories.

### Generation Modes & Sub-Batch Processing

- **Splitting by Promoter Constraint:**  
  If multiple `fixed_elements` constraints are specified in the YAML configuration, **densegen** automatically splits each input source into sub‐batches, ensuring that a unique combination of input source and promoter constraint is handled with its own quota and output directory.

- **Round-Robin vs. Sequential Modes:**  
  - **Sequential Mode:** Processes each sub-batch to completion before moving on.
  - **Round-Robin Mode:** When `round_robin` is set to `true`, **densegen** interleaves sequence generation across all sub-batches. This means it generates one (or a few) sequences per sub-batch in a cyclic fashion, ensuring balanced output even if the process stops early.

- **Limited Generation Calls:**  
  A helper function limits the number of sequences generated per call (using a `max_sequences` parameter), which supports round-robin interleaving without overloading any single sub-batch.

### Module Overview

- **`main.py`:**  
  The CLI entry point that loads configurations, coordinates data ingestion, manages sub-batch splitting, and triggers sequence generation.

- **`data_ingestor.py`:**  
  Contains classes for loading CSV or PyTorch data. Supports various source types from the deg2tfbs repository (pipeline fetcher or cluster analysis).

- **`sampler.py`:**  
  Implements routines to randomly sample transcription factors and binding sites, ensuring diversity (with an option for unique TF-only sampling).

- **`optimizer_wrapper.py`:**  
  Extends the dense-arrays Optimizer, adding functionalities like fixed element constraints and gap-filling for achieving the desired sequence length and GC content.

- **`progress_tracker.py`:**  
  Tracks sequence generation progress in YAML files, allowing **densegen** to resume processing and maintain batch status.

- **`utils.py`:**  
  Provides shared utilities such as path resolution, configuration loading, and standardized sequence saving.

---

**densegen** is designed to be modular, extensible, and robust—facilitating high-throughput design of synthetic promoter sequences with tight regulatory element control. For any questions or contributions, please refer to the repository documentation.