# Welcome to dnadesign (DNA Regulatory Sequence Design)

This directory contains a collection of Python modules and data analysis pipelines, all related to non-coding DNA sequence design during Eric J. South's PhD research at Boston University.

## Installation

1. *(Optional but recommended)* Create an empty conda environment:
   ```bash
   conda create -n envname
   conda activate envname
   ```

2. Below is an example sequence of shell commands to create and configure your environment. Adapt Python versions or dependency versions as needed:

   ```bash
   # (Optional) Install mamba into your environment to speed up dependency resolution and installation.
   conda install -c conda-forge mamba -y

   # Install PyTorch, TorchVision, and TorchAudio from pytorch and nvidia channels.
   mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
   
   # Note that pytorch-cuda and nvidia are required for running Evo, which uses FlashAttention. 
   # pytorch-cuda=11.8 is specified here, but you may need to adjust to match your system's GPU capabilities.
   
   # Install ScanPy 1.10.3, seaborn, numpy, pandas, etc.
   mamba install scanpy=1.10.3 seaborn numpy pandas matplotlib -c conda-forge -y

   # Install the evo-model package via pip
   pip install evo-model
   ```   
   See Evo's installation documentation [here](https://github.com/evo-design/evo/tree/main) for more context.

## Directory Structure

1. **preprocessing**
([more **preprocessing** documentation here](preprocessing/preprocessing-docs.md))
   - Contains Python modules and scripts used to ingest external data and prepare for downstream pipelines in sibling project directories.
   - These data sources include:
     - Comparative RNA-seq and proteomics datasets derived from *E. coli* studies in the literature.
     - Bacterial promoter engineering datasets.
     - Promoter and transcription factor binding site tables from **RegulonDB** (releases 11 and 12) and **EcoCyc** (releases 27 and 28).

2. **sequences**
([more **sequences** documentation here](preprocessing/sequences-docs.md))
   - This directory contains sequences outputted from **preprocessing** and tidied into a standardized data structure. Each sequence entry includes:
     - A **unique identifier** (id).
     - The molecular (DNA) **sequence**.
     - **Metadata** describing the sequence, such as:
       - Type (e.g., promoter or binding site).
       - Descriptive labels (e.g., ascribed categories or numerical values, such as experimentally observed transcription strength).
   - Example data structure:
     ```python
      my_sequences = [
         {
            "id": None,
            "sequence": None,
            "meta_type": None,
            "meta_source": None,
            "meta_date_accessed": None,
            "meta_regression_label": None,
            "meta_category_label": None,
            "tensor": None,
         },
         # Add additional dictionaries here to extend the list
         ]
     ```
   - **Note:** To process your own set of sequences through downstream modules, format your data as a list of dictionaries matching the example structure shown above, and save it as a ```.pt``` file.

3. **evoinference**
   - This directory passes sequences through **[Evo](https://github.com/evo-design/evo/tree/main)** (checkpoint name: `evo-1.5-8k-base`), a DNA foundation model for molecular-to-genome-scale modeling and design.
   - **evoinference** generates rich embeddings for sequences produced by **preprocessing**, ultimately adding an `evo_`-prefixed key followed by a tensor embedding.
   - Example data structure:
     ```python
     template = {
         "id": None,
         "sequence": None,
         "meta_type": None,
         ...
         "evo_output_head": None,
         "evo_output_head_mean_pooled": None,
     }
     ```
   - More information on Evo and semantic mining can be found in these papers:
     - [Sequence modeling and design from molecular to genome scale with Evo](https://www.science.org/doi/10.1126/science.ado9336)
     - [Semantic mining of functional de novo genes from a genomic language model](https://doi.org/10.1101/2024.12.17.628962)

4. **someclustering**
   - [Placeholder for description]

5. **archived**
   - Drafts of earlier packages, such as *decoydesigner*, *seqcnn*, and *densepromoters*, along with their associated preprocessing modules, are preserved here for reference.
   - Some of these archived directories include hard-coded paths that expect a 'preprocessing' directory to be a sibling, so maintaining the directory structure is essential for functionality.

6. **prototypes**
   - Chicken scratch!

---

_Last updated: 2025-01-23_

