# dnadesign

This repository contains a collection of Python modules and bioinformatic pipelines related to DNA sequence design.

## Installation

### Usage Style 1: Local Machine (No Gurobi or CUDA Support)

This style is appropriate for workflows that ***do not*** require heavy [dense array](https://gitlab.com/dunloplab/dense-arrays) computations or [Evo 2](https://github.com/ArcInstitute/evo2) inference.

1. Create and Activate a Conda Environment

   ```bash
   conda create -n dnadesign_local python=3.11 -y
   conda activate dnadesign_local
   ```

2. Install Dependencies

   ```bash
   conda install pytorch torchvision torchaudio scanpy=1.10.3 seaborn numpy pandas matplotlib pytest pyyaml -c conda-forge -y
   ```

3. Clone and Install the `dnadesign` Repository

   ```bash
   git clone https://github.com/e-south/dnadesign.git
   cd dnadesign
   pip install -e . # Install the local dnadesign package in editable mode
   ```
   *Installing in editable mode ensures that changes to the source files are immediately reflected without needing a reinstall.*

4. (Optional) Clone the [`dense-arrays`](https://gitlab.com/dunloplab/dense-arrays) Package

   The **densegen** workflow relies on the dense-arrays package. Install it as a sibling directory to `dnadesign`.
   ```bash
   git clone https://gitlab.com/dunloplab/dense-arrays.git
   cd dense-arrays
   pip install .
   cd ..
   ```
   
---

### Usage Style 2: Shared Computing Cluster (With Gurobi/CUDA Support)
This setup is designed for running more resource-intensive workflows on a [shared computing cluster](https://www.bu.edu/tech/support/research/system-usage/connect-scc/scc-ondemand/), such as solving dense array with [Gurobi](https://www.gurobi.com/), or performing inference with [Evo 2](https://github.com/ArcInstitute/evo2). For Evo 2’s FP8 features, a GPU with compute capability **8.9 or higher** is required.

> Interactive Session Resource Request Example:
> - **densegen** workflow:
>   - Modules: miniconda gurobi
>   - Cores: 16  
>   - GPUs: 0  
> - **evoinference** workflow:  
>   - Modules: cuda miniconda
>   - Cores: 3  
>   - GPUs: 1  
>   - GPU Compute Capability: 8.9  
>   - Extra options: `-l mem_per_core=8G`  
>   
> (Check your cluster documentation for submission details.)

1. Set Up the CUDA Environment

   Evo 2’s GPU-accelerated components require NVIDIA’s CUDA toolkit. This step loads the necessary CUDA and GCC modules, verifies the presence of the CUDA compiler (nvcc), and exports environment variables so that both Python and build scripts can locate the CUDA installation. These settings are crucial for compiling CUDA extensions and ensuring compatibility with PyTorch.

   ```bash
   module load cuda/12.5      # Load the CUDA module appropriate for your cluster
   module load gcc/10.2.0     # Load a GCC version that is compatible with CUDA

   # Verify that nvcc is available:
   ls $CUDA_HOME/bin/nvcc  # This should display the path to the nvcc binary

   # Export CUDA environment variables:
   export CUDA_HOME=/share/pkg.8/cuda/12.5/install
   export CUDA_PATH=/share/pkg.8/cuda/12.5/install
   export CUDA_TOOLKIT_ROOT_DIR=/share/pkg.8/cuda/12.5/install
   export CUDA_BIN_PATH=/share/pkg.8/cuda/12.5/install/bin
   export PATH=$CUDA_BIN_PATH:$PATH
   export NVCC=$CUDA_BIN_PATH/nvcc

   # Optional: Check versions to verify correct module load and installation
   nvcc --version
   gcc --version
      ```

2. Create and Activate the Conda Environment

   ```bash
   conda create -n dnadesign_cu126 python=3.11 -y
   conda activate dnadesign_cu126
   ```

3. (Optional) Install Mamba and Upgrade Build Tools

   Mamba speeds up dependency resolution and installation. Upgrading build tools (pip, setuptools, wheel) just ensures compatibility and access to the latest features.
   ```bash
   conda install -c conda-forge mamba -y
   unset -f mamba  # (Optional: Unset mamba shell function if conflicts occur)
   mamba install pip -c conda-forge -y
   pip install --upgrade pip setuptools wheel
   ```

4. Install PyTorch with CUDA Support

   Installing PyTorch built for CUDA ensures that GPU acceleration is enabled for Evo 2’s computations. Here, we install a version built for CUDA 12.6, which is optimal for GPUs with compute capability ≥8.9.
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
   ```
   > **Note:** If your GPU does not support FP8 or if you encounter compatibility issues, consider installing a version built for an older CUDA (e.g., cu118) and try and adjust Evo 2’s configuration.

5. Install Additional Packages via Mamba

   These scientific and plotting libraries are required by various subprojects within dnadesign.
   ```bash
   mamba install scanpy=1.10.3 seaborn numpy pandas matplotlib pytest pyyaml -c conda-forge -y
   ```

6. Install **Evo 2**

   Cloning with submodules ensures that all dependencies, including those in external repositories, are included.
   ```bash
   git clone --recurse-submodules git@github.com:ArcInstitute/evo2.git
   cd evo2
   ```

7. Override CUDA Paths in the Makefile (if necessary)

   If Evo 2’s build system does not detect your CUDA installation correctly, update the Makefile in the vortex directory to use the correct paths:

   ```makefile
   # Change the ":=" to "?=" for these lines
   CUDA_PATH ?= /usr/local/cuda
   CUDA_HOME ?= $(CUDA_PATH)
   CUDACXX ?= $(CUDA_PATH)/bin/nvcc
   ```
   Change these defaults so that your exported environment variables take precedence.


8. Install Evo2 in Editable Mode
   ```bash
   cd evo2
   pip install -e .
   ```

9. (Optional) Build Additional Components

   Some Evo 2 features, such as custom CUDA extensions, require a build step. Running make setup-full compiles these extensions.
   ```bash
   cd vortex
   make setup-full CUDA_PATH=/share/pkg.8/cuda/12.5/install CUDACXX=/share/pkg.8/cuda/12.5/install/bin/nvcc CUDA_HOME=/share/pkg.8/cuda/12.5/install
   cd ..
   ```

10. Test the Evo 2 Installation
   
      Running a test script verifies that the installation was successful and that Evo2 can access the necessary resources and configurations.
      ```bash
      cd evo2
      python ./test/test_evo2.py --model_name evo2_7b
      ```

11. (Optional) Clone the [`dense-arrays`](https://gitlab.com/dunloplab/dense-arrays) Package

      The **densegen** workflow relies on the dense-arrays package. Install it as a sibling directory to `dnadesign`.
      ```bash
      git clone https://gitlab.com/dunloplab/dense-arrays.git
      cd dense-arrays
      pip install .
      cd ..
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
