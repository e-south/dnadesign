## densehairpins

**densehairpins** is a pipeline for short dense-array design and analysis. It processes transcription factor binding site datasets, iteratively subsamples among these sequences and runs a [**dense-array**](https://github.com/e-south/dense-arrays) solver to generate short compound sequences, scores these solutions based on multiple metrics, and visualizes the results.

### Usage

1. **Configure the Pipeline**  
   Edit the YAML file (e.g., `configs/example.yaml`) under the `densehairpins` key to specify parameters such as analysis style, solver options, score weights, and subsampling settings.

2. **Prepare Input Data**  
   Ensure that input motif files (in MEME format) and required datasets (e.g., the Pancardo dataset) are available as specified in the configuration and associated with the proper paths.

3. **Run the Pipeline**  
   From the `densehairpins` directory, execute:
   ```bash
   python main.py
   ```
   This will:
   - Parse and deduplicate binding site entries.
   - Create a batch output folder (with subdirectories for CSVs and plots).
   - Iteratively run the solver to generate candidate solutions.
   - Score and rank the solutions.
   - Produce visualizations (e.g., scatter plots) and export results as CSV files.

4. **Outputs**  
   Check the generated batch folder under `dnadesign/densehairpins/batch_results/` for:
   - **CSV files** (e.g., ranked solutions, intermediate data)
   - **Plots** (e.g., scatter plots of cumulative score vs. sequence length)
   - **Progress Status** stored as YAML files
