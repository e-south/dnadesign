## tfkdanalysis

**tfkdanalysis** is a pipeline for analyzing transcription factor knockdown (TFKD) effects using data from **PPTP-seq (Promoter responses to TF perturbation sequencing)**. It processes promoter activity responses from CRISPRi-based TF knockdowns in *E. coli* and generates volcano and scatter plots to highlight significantly affected genes under specified conditions.

This pipeline uses data from:

> **Han et al., 2023**  
> *Genome-wide promoter responses to CRISPR perturbations of regulators reveal regulatory networks in Escherichia coli*  
> **DOI:** [10.1038/s41467-023-41572-4](https://doi.org/10.1038/s41467-023-41572-4)  
> - 183 TF genes knocked down  
> - 1372 *E. coli* promoters measured  
> - 200,000 TF-gene response profiles  
> - Data from **Supplementary Data 6** (PPTP-seq interactions) and **7** (known interactions)

\
**tfkdanalysis** loads these datasets, selects transcription factors of interest (as specified in the config), filters by media condition, and visualizes genes that are significantly up- or down-regulated in the knockdown context.

### Usage

1. **Edit the Config**  
   Configure your batch in `configs/example.yaml`, including:
   ```yaml
   tfkdanalysis:
     batch_name: "araC_and_pspF"
     regulators: ["araC", "pspF", "marR", ...]
     media: "glu"
     volcano_plot: true
     threshold: 1.2
     annotate_operon: true
   ```

2. **Run the Pipeline**
   ```bash
   python main.py
   ```

3. **View Outputs**  
   Results are saved under `batch_results/<timestamp>_<batch_name>_<regulators>/`, including:
   - **Plots/**: Volcano and scatter plots
   - **CSVs/**: Up/down-regulated genes per regulator
