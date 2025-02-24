# dnadesign

This repository contains a collection of Python modules and bioinformatic pipelines related to DNA sequence design.

### Directory Layout
```text
dnadesign/
├── README.md                           # High-level project documentation
└── src/
    └── dnadesign/
        ├── configs/
        │   └── example.yaml            # Global configuration for all pipelines
        ├── utils.py                    # Shared utilities (e.g., config loading, common functions)
        ├── seqfetcher/                 # Data ingestion modules (one per dataset)
        │   └── <dataset>_module.py     
        ├── densegen/                 
        │   ├── main.py                 # CLI entry point for densegen pipeline
        │   └── ...  
        ├── sequences/                  
        │   ├── seqmanager.py           # Tool for validating and inspecting .pt files
        │   └── seqbatch_<name>/        # Each subdirectory contains:
        │       ├── <batch>.pt          # Torch file with a list-of-dicts (each dict represents a sequence)
        │       └── summary.yaml        # YAML summary of the batch (metadata, parameters, runtime)
        ├── evoinference/              
        │   ├── main.py                 # CLI entry point for evoinference pipeline
        │   └── ...              
        └── clustering/                
            ├── main.py                 # CLI entry point for clustering pipeline
            └── ...                  
                  
```
     
### Pipelines

1. [**seqfetcher**](src/dnadesign/seqfetcher/README.md)
   
   **seqfetcher** is a data ingestion pipeline that is designed to reference a sibling directory [**dnadesign-data**](https://github.com/e-south/dnadesign-data), which includes bacterial promoter engineering datasets curated from primary literature, along with experimental datasets detailing other promoters and transcription factor binding sites derived from [**RegulonDB**](https://regulondb.ccg.unam.mx/) and [**EcoCyc**](https://ecocyc.org/).

2. [**densegen**](src/dnadesign/densegen/README.md) 
   
   **densegen** is a DNA sequence design pipeline built on the integer linear programming framework from the [**dense-arrays**](https://github.com/e-south/dense-arrays) package. It assembles batches of synthetic promoters with densely packed transcription factor binding sites. The pipeline references curated datasets from the [**deg2tfbs**](https://github.com/e-south/deg2tfbs) repository, subsampling dozens of binding sites for the solver while enforcing time limits to prevent stalling.

3. [**sequences**](src/dnadesign/sequences/README.md)
 
   **sequences** serves as the central storage location for nucleotide sequences within the project, organizing and updating outputs from **seqfetcher**, **densegen**, and **evoinference** into a standardized data structure. Subdirectories are prefixed with *seqbatch* or *densebatch* to indicate their source and contain both `.yaml` files, which provide batch summaries, and a corresponding `.pt` file storing sequences and metadata. Each sequence file is structured as a list of dictionaries, following this format:  
   ```python
   example_sequence_entry = [

      {
         "id": "90b4e54f-b5f9-48ef-882a-8763653ae826",
         "meta_date_accessed": "2025-02-19T12:01:30.602571",
         "meta_source": "deg2tfbs_all_DEG_sets",
         "sequence": "gtactgCTGCAAGATAGTGTGAATGACGTTCAATATAATGGCTGATCTTATTTCCAGGAAACCGTTGCCACA",
         "meta_type": "dense-array",
         "evo2_logits_mean_pooled": tensor([[[-10.3750,  10.3750, ..., 10.3750,  10.3750]]], dtype=torch.bfloat16),
         "evo2_logits_shape": [1, 512]
      },
      # Additional dictionary entries extend the list
   ]
   ```
   **Note:** To process custom sequences through downstream modules, format your data as a list of dictionaries matching the structure above and save it as a `.pt` file.

4. [**evoinference**](src/dnadesign/evoinference/README.md)  

   **evoinference** is a wrapper for **[Evo 2](https://github.com/ArcInstitute/evo2)** (checkpoint: `evo2_7b`), a genomic foundation model for molecular-to-genome-scale modeling and design. This pipeline processes batches of `.pt` files from the sibling **sequences** directory, passing each sequence through Evo 2 and extracting tensors derived from output logits or intermediate layer embeddings from the LLM. The extracted data is then saved in place as additional keys within the original `.pt` file.  

   For more context, see the following papers:  

   - [Semantic mining (preprint)](https://doi.org/10.1101/2024.12.17.628962)  
   - [Evo 1](https://www.science.org/doi/10.1126/science.ado9336)  
   - [Evo 2 (preprint)](https://arcinstitute.org/manuscripts/Evo2)


4. [**clustering**](src/dnadesign/clustering/README.md) 
  
   **clustering** utilizes [Scanpy](https://scanpy.readthedocs.io/en/stable/) for cluster analysis on nucleotide sequences stored in the sibling **sequences** directory. By default, it uses the mean-pooled output logits of **Evo 2** along the sequence dimension as input. The pipeline generates UMAP embeddings, applies Leiden clustering, and also supports downstream analyses, such as cluster composition and diversity assessment.


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
   conda install pytorch torchvision torchaudio scanpy=1.10.3 seaborn numpy pandas matplotlib pytest pyyaml leidenalg igraph openpyxl -c conda-forge -y
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
   mamba install scanpy=1.10.3 seaborn numpy pandas matplotlib pytest pyyaml leidenalg igraph openpyxl -c conda-forge -y
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


8. Install Evo 2 in Editable Mode
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
      ```

## **Usage Example**

1. Clone the [**dnadesign-data**](https://github.com/e-south/dnadesign-data) repository, and place it as a sibling directory to **dnadesign**. This will enable **seqfetcher** to generate custom lists of dictionaires from these sources. 

2. Update the `configs/example.yaml` file as desired and try running different pipelines.
