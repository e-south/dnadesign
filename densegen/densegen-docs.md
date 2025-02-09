## densegen – Dense Array Generator
**densegen** is a bioinformatics pipeline for batch assembly of synthetic bacterial promoters with optimally packed transcription factor binding sites.

**Data Source:**  
densegen uses CSV data from the [**deg2tfbs**](https://github.com/e-south/deg2tfbs/tree/main) repository. These CSV files contain curated sets of transcription factors and their corresponding binding sites.

**Optimization Engine:**  
The pipeline wraps around integer linear programming solvers described in the [**dense-arrays**](https://github.com/e-south/dense-arrays) project.

**Output:**  
Generated sequences, along with metadata, are saved as PyTorch `.pt` files in a structured batch directory.

---
#### Directory Layout

```bash
dnadesign/
├── __init__.py               
├── setup.py
├── config/
│   └── config.yaml            # Central configuration file
├── sequences/                 
├── utils.py                   # Central utilities (paths, constants, etc.)
└── densegen/
    ├── __init__.py            
    ├── config_loader.py       # Loads configuration from YAML.
    ├── data_ingestor.py       # Ingests CSV data from the deg2tfbs package.
    ├── sampler.py             # Groups data by TF and subsamples binding sites.
    ├── optimizer_wrapper.py   # Wraps dense_arrays.Optimizer with retry/fail-safe.
    ├── progress_tracker.py    # Tracks overall progress to a YAML file.
    ├── sequence_saver.py      # Saves the output (list of dicts) to file.
    └── main.py                # Main entry point that orchestrates the process.
```

---

### Pipeline Workflow
*TFBS Library → Subset Sampling → SPP Optimization → dsDNA Output*

1. **Configuration Loading**  
   - **Module:** `config_loader.py`  
   - **Function:** Reads a central YAML configuration (e.g. `configs/example.yaml`) to set parameters such as input/output directories, solver preferences, sequence length, number of sequences to generate (i.e., 'quota'), and fixed constraints.

2. **Data Ingestion**  
   - **Module:** `data_ingestor.py`  
   - **Function:** Loads CSV files from a deg2tfbs data directory, validating required columns and removing duplicates or empty entries.  


3. **Sampling**  
   - **Module:** `sampler.py`  
   - **Function:** Performs subsampling on ingested TF-TFBS tables. Supports unique-TF sampling and random selection with replacement if necessary.


4. **Optimization**  
   - **Module:** `optimizer_wrapper.py`  
   - **Function:** Wraps the `dense_arrays.Optimizer` from the dense-arrays project, applying any fixed constraints (e.g., promoter or side biases) and gap-fill options to the generated sequences.


5. **Progress Tracking and Output Saving**  
   - **Modules:** `progress_tracker.py` and `sequence_saver.py`  
   - **Function:**  
     - **Progress Tracking:** Writes progress updates to a YAML file, logging details such as the number of sequences generated and any errors encountered.
     - **Output Saving:** Saves the list of generated sequence entries (each a dictionary with metadata) to a PyTorch `.pt` file, uniquely named by source and timestamp.


6. **Pipeline Overview**  
   - **Module:** `main.py`  
   - **Function:** Runs the full process. For each source defined in the configuration, it:
     - Loads and validates data.
     - Samples binding sites.
     - Generates promoter sequences via the optimizer.
     - Tracks progress and saves output.
     - Logs detailed runtime information such as elapsed time, TFs used, and a visual summary of each generated sequence.

---

## Usage

While installation details are provided elsewhere, to run the DenseGen pipeline, simply execute the main module:

```bash
python dnadesign/densegen/main.py
```

Ensure that your configuration file (e.g. `configs/example.yaml`) is correctly set up and that the paths to the deg2tfbs input directories are valid.