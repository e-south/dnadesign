## billboard

**billboard** performs analyses and calculates entropy metrics on batches of regulatory sequence data stored in its sibling **sequences** directory. **billboard** processes each sequence to extract transcription factor binding site (TFBS) information, computes various entropy metrics that describe both positional (global and per‑TF) and combinatorial diversity, and generates summary CSV files. Optionally, the program can also produce a set of plots to visualize TF frequency, occupancy, motif lengths, and TF statistics.

The program is configured via a YAML file (e.g., `dnadesign/configs/example.yaml`). A special **DRY_RUN** mode is available to quickly perform the analysis and produce only an entropy summary CSV file, without generating plots.

### Modules

- **`main.py`**

    The entry point that loads configuration settings, parses input `.pt` files, and directs the program flow (full run or DRY_RUN).

- **`core.py`** 
    
    Contains the core logic to extract and process transcription factor binding sites (TFBS) from sequences, build positional coverage matrices, and calculate key entropy metrics. 
    
    For every sequence, the module:
    - Parses each entry in `"meta_tfbs_parts"` to extract the TF name and its binding motif.
    - Searches for the motif in the full sequence and constructs two positional coverage matrices (one for the forward strand and one for the reverse) that record how many times motifs appear at each nucleotide position.

   The module then computes several types of entropy metrics to quantify the diversity and spatial distribution of motifs:
   
   **Global Positional Entropy:**  
     - *Definition:* The Shannon entropy of the motif coverage across all positions in the entire library of sequences.  
     - *Computation:* Aggregates counts from the forward and reverse coverage matrices, computes the entropy, averages the two, and then normalizes by dividing by log₂(sequence length).  
     - *Interpretation:* Indicates whether motifs are clustered in specific regions (low entropy) or evenly distributed across the sequence (high entropy).

   **Per‑TF Positional Entropy:**  
     - *Definition:* The entropy calculated for the positional distribution of a specific transcription factor's binding sites.  
     - *Computation:* For each TF, the entropy is computed from its forward and reverse coverage, averaged, and normalized by log₂(sequence length).  
     - *Interpretation:* High per‑TF entropy suggests that a TF binds over many positions (flexible binding), while low entropy indicates restricted binding locations. Summary statistics (unweighted mean, frequency‑weighted mean, and top‑K average) can be derived from these values.
     
    **TF Frequency Entropy:**  
    - *Definition:* A measure of how uniformly transcription factors (TFs) are represented across the library.  
    - *Computation:* Convert the frequency counts of each TF into a probability distribution and compute the Shannon entropy, then normalize by dividing by log₂(the number of unique TFs).  
    - *Interpretation:* Higher entropy indicates a more even, egalitarian distribution of TF frequencies; lower entropy suggests that a few TFs dominate the library.

    Together, these calculations offer a ensemble view of regulatory sequence diversity, detailing both the global pattern of motif distribution and the individual behavior of transcription factors.

- **plot_helpers.py**  
  
  Provides functions to generate diagnostic plots (TF frequency bar plot, occupancy heatmap, motif length density plot, etc.).

- **summary.py**  

  Contains functions to generate CSV outputs. There are two CSV generation paths:  
  - **Full Run**: Outputs multiple CSVs including summary metrics, per‑TF coverage, mapping failures, TF combination counts, and an overall entropy summary.  
  - **DRY_RUN Mode**: Generates only the entropy summary CSV.

## Simple Usage

- **Configuration:**  
   Edit the `configs/example.yaml` file to set the desired configuration options.

- **Execution:**  
   Run the main program using:
   ```bash
   python billboard/main.py
   ```
   The program will:
   - Load the configuration and input `.pt` files.
   - Process sequences and compute metrics.
   - Create an output directory under `batch_results/<output_dir_prefix>_<date>`.
   - Generate CSV files (and plots if not in DRY_RUN mode).