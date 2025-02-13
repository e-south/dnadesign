## densegen – Dense Array Generator

**densegen** is a DNA sequence design pipeline, wrapped around the integer linear programming package described in the [**dense-arrays**](https://github.com/e-south/dense-arrays), for batch assembly of synthetic bacterial promoters with densely packed transcription factor binding sites. **densegen** reads ```tf2tfbs_mapping.csv``` files from the [**deg2tfbs**](https://github.com/e-south/deg2tfbs/tree/main) repository, which details curated sets of transcription factors and their corresponding binding sites. Generated dense arrays and their metadata are saved as PyTorch `.pt` files within a structured batch directory.

#### Directory Layout

```python
dnadesign/
├── __init__.py               
├── config/
│   └── example.yaml           # User-defined configurations.
├── utils.py                   # Central utilities (paths, constants, etc.).
├── densegen/
│   ├── __init__.py            
│   ├── main.py                # CLI entry point.
│   ├── data_ingestor.py       # Ingests CSV data from the deg2tfbs package.
│   ├── sampler.py             # Randomly samples TFs, then randomly selects a binding site from each.
│   ├── optimizer_wrapper.py   # Extends the dense_arrays.Optimizer class.
│   ├── progress_tracker.py    # Tracks batch sequence generation in a YAML file.
│   └── batches/
│       └── seqbatch_<id>.pt   # Timestamped directory storing generated sequences.
└── sequences/                  
```

### Simple Usage

1. Clone the [**deg2tfbs**](https://github.com/e-south/deg2tfbs) repository to access a curated set of ```tf2tfbs_mapping.csv``` files. Placing it as a sibling directory to **dnadesign** enables **densegen** to generate dense arrays from these sources.

2. Update ```mycustomparams.yaml``` with the desired I/O paths, batch IDs, and dense array design preferences. For instance:

```yaml
# densegen/configs/mycustomparams.yaml

densegen:
  input_dir: "pipeline/tfbsfetcher"                 # Relative to deg2tfbs package root.
  output_dir: "sequences"                           # Defines where to save outputs.
  progress_file: "progress_status.yaml"             # Minimal progress tracker file.
  solver: "GUROBI"                                  # Preferred solver; falls back to CBC if necessary.
  solver_options:
    - "Threads=16"
    - "TimeLimit=20"
  sequence_length: 100                              # Defines the length of each dense array.
  quota: 5                                          # Total number of dense arrays to generate.
  subsample_size: 10
  arrays_generated_before_resample: 1               # Set >1 to generate multiple arrays per TF-TFBS sample before resampling.
  sources:                                          # Directory name references for data in deg2tfbs. 
    - "tfbsbatch_20250130_heat_shock_up"
    - "tfbsbatch_20250130_nutrient_lim_up"
    - "tfbsbatch_20250130_poor_carbon_acetate_up"
  fixed_elements:                                   # Incorporate positional constraints into solutions.
    promoter_constraints:
      - upstream: "TTGACA"
        downstream: "TATAAT"
        upstream_pos: [20, 80]
        spacer_length: [16, 18]
    side_biases:
      left: []
      right: []
  unique_tf_only: true                              # Enforce one binding site per TF per sequence.
  fill_gap: true                                    # Padding with filler sequences to match the desired length.
  fill_gap_end: "5prime"     
  fill_gc_min: 0.40
  fill_gc_max: 0.60
```

Then execute the main module:

```bash
python dnadesign/densegen/main.py
```

For each ```tf2tfbs_mapping.csv``` file defined in the configuration, it:
1. Loads and validates data.
2. Samples binding sites.
3.  Generates promoter sequences via the optimizer.
4.  Tracks progress and saves output.




