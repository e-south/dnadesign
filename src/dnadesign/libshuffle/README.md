## libshuffle Specification

This document details the design and functionality of the **libshuffle** module,
which performs iterative random subsampling of sequence libraries (stored as a `.pt` file),
computes diversity metrics (Billboard and Evo2), and visualizes and outputs the results.

## Overview
```bash
dnadesign/
├── README.md
└── src/
    └── dnadesign/
        ├── configs/
        │   └── example.yaml         # Global configuration (see below)
        ├── libshuffle/              # New module for libshuffle
        │   ├── __init__.py          # (empty)
        │   ├── config.py            # Loads libshuffle configuration
        │   ├── metrics.py           # Computes billboard and evo2 diversity metrics
        │   ├── subsampler.py        # Handles subsampling, deduplication, caching, and progress
        │   ├── visualization.py     # Generates the scatter plot visualization
        │   ├── output.py            # Handles saving of YAML summaries, results, and selected pt files
        │   └── main.py              # CLI entry point for libshuffle
        ├── sequences/               # (contains .pt files, etc.)
        └── ...                      # Other pipelines (densegen, evoinference, etc.)
```

- **Input Resolution:**  
  The user supplies a directory (e.g., `sequences/seqbatch_example/`) containing exactly one `.pt` file.
  The module verifies the presence of unique `"id"` fields in each sequence entry.

- **Subsampling & Metrics:**  
  For a configurable number of draws (default 200), a fixed-size subsample (default 100) is randomly drawn.
  Two metrics are computed:
  - *Billboard Diversity:* Simulated by summing entropy-related values.
  - *Evo2 Diversity:* Computed as the mean pairwise L2 (or alternative) distance between latent vectors.

- **Caching & Deduplication:**  
  Previously computed subsample metrics are cached using a frozenset of sequence IDs.
  Duplicate subsamples trigger resampling (up to a maximum number of attempts).

- **Joint Selection:**  
  If enabled, subsamples are evaluated against minimum threshold values for both metrics.
  The top-k subsamples may be saved separately for downstream analysis.

- **Visualization:**  
  A scatter plot is generated (Evo2 metric on the x-axis, Billboard metric on the y-axis),
  using Seaborn’s "ticks" style. Selected subsamples are highlighted.

- **Output:**  
  Summary (`summary.yaml`) and results (`results.yaml`) files are written along with the scatter plot.
  Optionally, selected subsample `.pt` files are saved back into the sequences directory.

## Implementation Notes

- Uses `pathlib` for path handling, `torch` for .pt file operations,
  and `tqdm` for progress reporting.
- Logging is handled via Python’s built-in `logging` module.
- The code is fully configuration-driven via a YAML file.
- Designed to be modular and easily extended in future iterations.

For complete details, please review the source code in this directory.
