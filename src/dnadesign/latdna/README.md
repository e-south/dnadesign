

## latdna (latent space analysis of DNA)

![latdna banner](assets/latdna-banner.svg)

**latdna** is a pipeline for latent space analysis of DNA sequences. It compares latent space diversity (via Evo2 embeddings) between dense array batches and synthetic (*latdna*) batches.

```bash
project_root/
├── configs/
│   └── example.yaml         # Central configuration file
├── latdna/
│   ├── __init__.py
│   ├── main.py              # Entry point: dispatches to generation or analysis
│   ├── generation.py        # Synthetic sequence generation pipeline
│   ├── analysis.py          # Analysis pipeline for latent diversity metrics
│   ├── metrics.py           # Metric functions (cosine, euclidean, log1p_euclidean)
│   ├── validation.py        # Assertive validation for data and config
│   └── utils.py             # Utility functions (random sequence generation, file I/O, etc.)
├── sequences/               # Directory for sequence batches
│   ├── densebatch_m9_acetate_tfs_n10000/
│   │   └── densebatch_m9_acetate_tfs_n10000.pt
│   └── latdnabatch_YYYYMMDD/  # Synthetic sequence outputs are stored here
│       ├── latdnabatch_YYYYMMDD.pt
│       └── generation_summary.yaml
└── latdna/batch_results/     # Analysis outputs
    └── latbatch_YYYYMMDD/
        ├── diversity_summary.csv
        ├── latent_diversity_boxplot.png
        └── analysis_config_snapshot.yaml
```

**latdna** has two modes:
1. **Generation Mode:**

    Create synthetic DNA sequences by tiling TF motifs into a reproducible random background with fixed sigma factor recognition site elements (i.e., the *latdna* batches).

2. **Analysis Mode:**

    Compute intra-population latent diversity metrics (cosine, Euclidean, log1p_Euclidean) and generate grouped boxplots comparing the dense array and latDNA batches.

### Configuration

All parameters are defined in a single YAML configuration file. Key configuration parameters include:
- **Mode Selection**:
  - Set latdna.mode to "generation" or "analysis" to choose the desired pipeline.
- **Generation Settings**:
  - dense_array_for_generation: Subdirectory name of the dense batch PT file.
  - sequence_length: Length of the synthetic sequences.
  - gc_content_range: Acceptable GC content range for the random background.
  - fixed_elements: Define upstream_seq, downstream_seq, upstream_start, and spacer_range.
  - tiling.step: Increment for motif tiling.
  - motifs: Map of TF names (case-insensitive) to one or more motif sequences. Duplicate motif sequences across different TFs are disallowed.
- **Analysis Settings**:
  - analysis_inputs: Specify the dense and latDNA batch subdirectory names.
  - metrics: List of metrics to compute (e.g., "cosine", "euclidean", "log1p_euclidean").
  - group_by: Either "all" (treat all latDNA entries as a single population) or "tf" (group by transcription factor).
- **Dry Run**:
  - Set latdna.dry_run to true to validate configurations without writing any output files.


### Usage

1. **Generation Mode**
    - Prepare the Dense Batch:
      - Ensure that the sequences/densebatch_m9_acetate_tfs_n10000/ directory exists and contains exactly one .pt file.
      - The PT file should be a list of dictionaries with the required keys:
sequence, sequence_length, fixed_elements, meta_tfbs_parts, and meta_tfbs_parts_in_array.
2. **Run Generation**:
    - Set latdna.mode to "generation" in configs/example.yaml.
    - Execute the main script:
      ```bash
      python -m latdna.main
      ```
    - The pipeline will:
	    - Generate a random "background" DNA sequence.
	    - Insert fixed sigma factor recognition sites at defined positions.
	    - Tile the specified motifs at valid positions (ensuring no overlap with fixed elements).
	    - For each valid tiling, create two entries (forward and reverse complement).
	    - Save the synthetic sequences as a single PT file in a new `latdnabatch_YYYYMMDD/` subdirectory under `sequences/`.

3. **Analysis Mode**
    - Confirm that the dense batch and a previously generated latDNA batch exist in the sequences/ folder.
    - Ensure that all entries in both PT files include the evo2_logits_mean_pooled key (this should be added by `evoinference`).
    - Run Analysis:
      - Set latdna.mode to "analysis" in configs/example.yaml.
      - Specify the subdirectory names for the dense and latDNA batches under latdna.analysis_inputs.
      - Execute the main script:
        ```bash
        python -m latdna.main
        ```
    - The pipeline will:
      - Extract latent embeddings from each entry.
      - Compute intra-population pairwise distances for each configured metric.
      - Generate a grouped boxplot for latent diversity and save it as latent_diversity_boxplot.png.
      - Save all analysis outputs in latdna/batch_results/latbatch_YYYYMMDD/.



To investigate how motif composition and transcription factor (TF) diversity are reflected in the latent space of Evo2, we generated dense promoter arrays composed of known TF binding sites and analyzed their intra-group pairwise distances in embedding space. The central question is whether combinations of binding sites associated with a higher diversity of TFs lead to greater intra-group dissimilarity. If so, this would suggest that the model is sensitive not just to sequence content, but also to combinatorial regulatory grammar — detecting not only the presence of binding sites but also their associated regulatory identities.

We compare three distinct categories of synthetic promoters:
	1.	Monotypic arrays contain only binding sites for a single transcription factor, using experimentally validated sites drawn from databases such as RegulonDB.
	2.	Restricted heterotypic arrays contain binding sites from a curated subset of transcription factors responsive to acetate transition. This represents a smaller regulatory vocabulary — a limited set of binding site signatures.
	3.	Unrestricted heterotypic arrays draw from all available TFs with binding site data, representing the broadest possible diversity and highest number of unique regulatory inputs.

By comparing intra-group distances across these categories, we aim to assess whether Evo2 embeddings reflect the underlying regulatory diversity encoded by binding site composition.
