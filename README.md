# dnadesign

This directory contains a collection of Python modules and bioinformatic pipelines, all related to DNA sequence design during Eric J. South's PhD research at Boston University.

## Installation

**Step 1.** Clone the repository
```bash
git clone https://github.com/e-south/dnadesign.git
cd dnadesign
```

**Step 2 (Option A).** Running Locally
```bash
# (Optional) Install mamba into your environment to speed up dependency resolution and installation.
conda install -c conda-forge mamba -y

# Install PyTorch, TorchVision, and TorchAudio from pytorch and nvidia channels.
mamba install pytorch torchvision torchaudio pytorch scanpy=1.10.3 seaborn numpy pandas matplotlib pytest pyyaml -c conda-forge -y
```   

Install a cloned [dense-arrays](https://github.com/e-south/dense-arrays) package via pip.
```bash
git clone https://gitlab.com/dunloplab/dense-arrays.git
cd dense-arrays
pip install .
```

Install the Local `dnadesign` Package in Editable Mode
```bash
(dnadesign) cd dnadesign
(dnadesign) pip install -e .
```
This allows Python to recognize **dnadesign** as an installed package while still linking directly to the source files in the repository. Any changes made to the source code will be immediately available without requiring reinstallation.

**Step 2 (Option B).** Running on a Shared Cluster (for using Evo)
```bash
git clone https://github.com/e-south/dnadesign.git
cd dnadesign

conda install -c conda-forge mamba -y

# Install PyTorch, TorchVision, and TorchAudio from pytorch and nvidia channels.
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install ScanPy 1.10.3, seaborn, numpy, pandas, etc.
mamba install scanpy=1.10.3 seaborn numpy pandas matplotlib pytest pyyaml -c conda-forge -y
```
*pytorch-cuda* and *nvidia* are required for running Evo, which uses FlashAttention. You may need to adjust *pytorch-cuda=11.8* to match your system's GPU capabilities.


Install the evo-model package via pip
```bash
pip install evo-model
```
See Evo's installation documentation [here](https://github.com/evo-design/evo/tree/main) for more context.

Install a cloned [dense-arrays](https://github.com/e-south/dense-arrays) package via pip.
```bash
git clone https://gitlab.com/dunloplab/dense-arrays.git
cd dense-arrays
pip install .
```

### Directory Layout
---
```text
dnadesign/
├── README.md
├── pyproject.toml
└── src/
    └── dnadesign/
        ├── __init__.py
        ├── utils.py
        ├── main.py                   # CLI entry point
        ├── configs/                  # User-defined configurations
        │   └── example.yaml          # Customize to process different DNA sequences
        ├── seqfetcher/
        │   ├── utils.py              # Contains paths to datasets (from dnadesign-data)
        │   ├── __init__.py 
        │   ├── <dataset>_module.py   # Each dataset has its own respective module
        │   └── ...  
        ├── densegen/
        │   ├── __init__.py 
        │   ├── <dataset>_module.py   # Each dataset has its own respective module
        │   └── ...  
        ├── sequences/                
        │   ├── __init__.py    
        │   ├── seqmanager.py         # Checks that incoming .pt files have proper shape
        │   └── seqbatch_<name>/      # Batch of sequences ingested from a given run
        │       ├── seqset_<name>.pt  # Data structure containing sequences
        │       ├── csvs                
        │       └── plots     
        ├── evoinference/            
        │   ├── __init__.py    
        │   ├── foo.py        
        │   └── bar/                
        │       ├── __init__.py
        │       └── ...                 
        └── clustering/            
            ├── __init__.py    
            ├── foo.py        
            └──  bar/                 
```
     
### Subdirectories

1. [**seqfetcher**](seqfetcher/seqfetcher-docs.md)
   - Loads bacterial promoter engineering datasets from the [**dnadesign-dna**](https://github.com/e-south/dnadesign-data) repository.
   - Loads curated experimental datasets, derived from [**RegulonDB**](https://regulondb.ccg.unam.mx/) or [**EcoCyc**](https://ecocyc.org/), also from [**dnadesign-dna**](https://github.com/e-south/dnadesign-data).
   - Loads sets of transcription factor binding sites

2. [**densegen**](densegen/densegen-docs.md) 
   - 

2. [**sequences**](sequences/sequences-docs.md)
   - This directory contains sequences outputted from **densegen** and tidied into a standardized data structure. Each sequence entry includes:
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

---

## **Usage Example**

1. Clone the [**dnadesign-data**](https://github.com/e-south/dnadesign-data) repository to access a curated set of various experimental datasets. Placing it as a sibling directory to **dnadesign** enables **preprocessing** to generate custom lists of dictionaires from these sources. 

---
