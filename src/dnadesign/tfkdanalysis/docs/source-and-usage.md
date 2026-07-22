# TFKDAnalysis Source and Usage

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14

TFKDAnalysis analyzes transcription factor knockdown effects from PPTP-seq
promoter-response data and renders volcano and scatter plots for configured
conditions.

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

### Configuration Contract

The checked-in configuration shape is illustrated by
`src/dnadesign/tfkdanalysis/config.yaml`:

   ```yaml
   tfkdanalysis:
     batch_name: "araC_and_pspF"
     regulators: ["araC", "pspF", "marR", ...]
     media: "glu"
     volcano_plot: true
     threshold: 1.2
     annotate_operon: true
   ```

### Runtime Status

There is no registered `tfkdanalysis` project script. The current `main.py`
uses a package-local import that is not module-safe and resolves the absent
`src/dnadesign/configs/example.yaml` rather than the checked-in package config.
Do not advertise `python main.py` as an operator path until those entrypoint and
configuration contracts are corrected.

When called from repaired orchestration, the analysis code writes under
`src/dnadesign/tfkdanalysis/batch_results/<date>_<batch_name>_<regulators>/`:

- `plots/`: volcano and regulator-scatter PNG files
- `csvs/`: up- and down-regulated rows per configured regulator
