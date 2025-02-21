# dnadesign

This directory contains a collection of Python modules and bioinformatic pipelines, all related to DNA sequence design during Eric J. South's PhD research at Boston University.

## Installation

### Usage Style 1: Local Machine with no Gurobi or CUDA support
For situations where you are not generating many dense arrays (see **densegen**) or making calls to Evo 2 (see **evoinference**).

**Step 1.** Clone the repository
```bash
git clone https://github.com/e-south/dnadesign.git
cd dnadesign
```

**Step 2.** Create Conda environment and install dependencies
```
conda create -n dnadesign_local python=3.11 -y
conda activate dnadesign_local
```
**Step 3.** Install dependencies
```
conda install pytorch torchvision torchaudio pytorch scanpy=1.10.3 seaborn numpy pandas matplotlib pytest pyyaml -c conda-forge -y
```   

**Step 4.** (Optional) If planning to use **densegen**, install a cloned [dense-arrays](https://github.com/e-south/dense-arrays) package via pip.
```bash
git clone https://gitlab.com/dunloplab/dense-arrays.git
cd dense-arrays
pip install .
```

**Step 5.** Install the Local `dnadesign` Package in Editable Mode
```bash
(dnadesign) cd dnadesign
(dnadesign) pip install -e .
```
*This allows Python to recognize **dnadesign** as an installed package while still linking directly to the source files in the repository. Any changes made to the source code will be immediately available without requiring reinstallation.*




















### Usage Style 2: Shared Computing Cluster with Gurobi or CUDA support
For situations where you want to leverage faster solver times with the Gurobi solver, or utilize NVIDIA GPUs with a CUDA Compute Capability of 8.9 which is needed for running Evo 2.

**Step 1.** When submitting your job or starting your interactive session, ensure your resource request specifies at least:
- **Number of cores:** 3  
- **Number of GPUs:** 1  
- **GPU Compute Capability:** 8.9  
- **Extra qsub options:** `-l mem_per_core=8G`

**Step 2.** Create and activate the Conda environment if you haven't already.
```bash
conda create -n dnadesign_cu126 python=3.11 -y
conda activate dnadesign_cu126
```
*Evo 2 requires CUDA support, so here we I indicate latest nvcc version in BU's shared computing cluster.*

**Step 3.** (Optional) Install Mamba to speed up dependency resolution.
   ```bash
   conda install -c conda-forge mamba -y
   # (Optional) Unset mamba shell function if it causes conflicts:
   unset -f mamba
   ```

3. Install pip via Mamba and upgrade build tools
   ```bash
   mamba install pip -c conda-forge -y
   pip install --upgrade pip setuptools wheel
   ```

4. Install PyTorch (with CUDA support)
   We are installing PyTorch built for CUDA 12.6 via the appropriate index (if you have GPU hardware with compute capability ≥8.9).  
   *Note: If your hardware does not support FP8, you might instead use a version built for an older CUDA (like cu118) and disable FP8 features in Evo2.*

   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
   ```

5. **Install Additional Packages via Mamba**

   ```bash
   mamba install scanpy=1.10.3 seaborn numpy pandas matplotlib pytest pyyaml -c conda-forge -y
   ```





### Interactive Session Requirements for Running Evo 2




1. **Set Up Your CUDA Environment Variables**

   Verify that `nvcc` is found in your CUDA installation, then export the necessary variables:

   ```bash
   module load cuda/12.5      # Ensure the proper CUDA module is loaded.
   module load gcc/10.2.0     # Load an appropriate GCC version.

   # Verify nvcc:
   ls $CUDA_HOME/bin/nvcc  # (This should return the path to nvcc.)

   # Now set the CUDA environment variables:
   export CUDA_HOME=/share/pkg.8/cuda/12.5/install
   export CUDA_PATH=/share/pkg.8/cuda/12.5/install
   export CUDA_TOOLKIT_ROOT_DIR=/share/pkg.8/cuda/12.5/install
   export CUDA_BIN_PATH=/share/pkg.8/cuda/12.5/install/bin
   export PATH=$CUDA_BIN_PATH:$PATH
   export NVCC=$CUDA_BIN_PATH/nvcc

   # (Optional) Check versions:
   nvcc --version
   gcc --version
   ```


7. **Install Evo2**

   First, clone Evo2 with its submodules, then install it in editable mode:

   ```bash
   git clone --recurse-submodules git@github.com:ArcInstitute/evo2.git
   cd evo2
   ```
Despite specifiying where your cuda install is by adding it tothe path, You may need to override where evo2 looks for your nvcc installation. this can be done by modiying the  Makefile in evo2/vortex/Makefile, and changing the following arguments

Modify the Makefile in the vortex directory
   ```make
   CUDA_PATH ?= /usr/local/cuda
   CUDA_HOME ?= $(CUDA_PATH)
   CUDACXX ?= $(CUDA_PATH)/bin/nvcc
```
*this will ensure the makefile though you’ve exported environment variables to point to*

hen execute pip install -e .
   pip install -e .
   

   *If you need to build additional components (e.g., the vortex extensions), you can run:*

   ```bash
   cd vortex
   make setup-full CUDA_PATH=/share/pkg.8/cuda/12.5/install CUDACXX=/share/pkg.8/cuda/12.5/install/bin/nvcc CUDA_HOME=/share/pkg.8/cuda/12.5/install
   cd ..
   ```









8. **Test the Installation**

   Run a provided test script (for example):

   ```bash
   python ./test/test_evo2.py --model_name evo2_7b
   ```

   *Note:* If FP8 execution is enabled by default but your GPU does not support it, you may need to disable FP8 in the model configuration.


















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
   - A DNA sequence design pipeline, wrapped around the integer linear programming package described in the **dense-arrays** package, for batch assembly of synthetic bacterial promoters with densely packed transcription factor binding sites. 

2. [**sequences**](sequences/sequences-docs.md)
   - This directory contains sequences outputted from **seqfetcher** and **densegen** and tidied into a standardized data structure. Each sequence entry includes:
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
