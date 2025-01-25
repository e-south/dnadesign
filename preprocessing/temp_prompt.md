I have a Python package called **`dnadesign`**, currently under development, which includes a directory called **`preprocessing`**. The goal is to adhere to pragmatic programming principles and software design best practices by restructuring the **`preprocessing`** directory and its codebase to be more robust, modular, intuitive, and maintainable. This restructuring may involve creating a separate package for 'DEGs_to_TFBSs' with a better, descriptive name if necessary.

### Context

1. **Purpose of `preprocessing`**:  
   The `preprocessing` directory processes disparate tabular data sources (e.g., `.csv`, `.tsv`, `.xlsx`) into a standardized list of dictionaries, which are saved as `.pt` files. These outputs are intended for use in the main `dnadesign` directory.

2. **Current Structure**:  
   The `preprocessing` directory contains the following:
   - `data/`: Raw data sources with diverse structures and formats.  
     Example structure:
     ```
     ├── EcoCyc_28/
     │   ├── ECOLI-regulatory-network.txt
     │   ├── SmartTable_All_Promoters.txt
     │   ├── SmartTable_All_Transcription_Factor_Binding_Sites.txt
     ├── RegulonDB_13/
     │   ├── PromoterSet.tsv
     │   ├── RegulonsSummaryData.csv
     │   ├── TF-RISet.tsv
     │   ├── ...
     ├── primary_literature/
     │   ├── Baba et al/
     │   ├── Baumgart et al/
     │   ├── ...
     ```
   - `src/`: Contains scripts like `data_loader.py` for tidying and standardizing raw data.

3. **Configurations**:  
   A separate `configs/` directory in the parent of preprocessing stores configuration files for different subprocesses in YAML format. Example:
   ```yaml
   # configs/evo_default.yaml
   data_sources:
     - name: ecocyc_promoters
       path: sequences/EcoCyc/ecocyc_28_promoters_2025-01-23.pt
     - name: regulondb_promoters
       path: sequences/RegulonDB/regulondb_13_promoters_2025-01-23.pt
   ```

4. **Challenges**:
   - The raw data sources are highly idiosyncratic, requiring custom tidying logic for each dataset.
   - The current implementation is consolidated into a single file (`data_loader.py`), which complicates scalability and maintenance.
   - The design does not fully adhere to principles of **design by contract**, **modularity**, and **decoupling**.

---

### Requirements for Restructuring

1. **Core Functionality**:
   - Implement a modular pipeline that:
     1. **Loads raw data** from diverse formats (e.g., `.csv`, `.tsv`, `.xlsx`, `.txt`).
     2. **Standardizes and tidies data**, handling idiosyncrasies specific to each source (e.g., varying column names, header rows).
     3. **Validates output** against a predefined schema.
     4. **Saves standardized data** as `.pt` files in a specified directory within `dnadesign`.

2. **Key Improvements**:
   - **Modularity**: Break down the monolithic `data_loader.py` into smaller, task-specific modules (e.g., file readers, data cleaners, validators).
   - **Decoupling**: Separate source-specific logic from the core preprocessing pipeline.
   - **Configuration Management**: Leverage YAML files for configurable parameters (e.g., source file paths, column mappings, validation rules).
   - **Intuitive Naming**: Ensure modules, functions, and any potential new package name are self-explanatory.
   - **Scalability**: Enable easy addition of new data sources and transformations without impacting existing functionality.

# fetcher

Many of the subfolders within the `primary_literature` folder (located in the `data` directory) are associated with a specific subprocess that involves analyzing comparative omics datasets. These datasets contain tabular data that need to be processed to identify **differentially expressed genes (DEGs)** by comparing two conditions in the dataset: a reference condition and a target condition. 

The goals are:
1. To adhere to pragmatic programming principles and software design best practices.
2. To restructure this directory and its codebase for improved robustness, modularity, intuitiveness, and maintainability. 
3. To enable the process to produce persistent outputs in the form of `.pdf` plots and well-structured results.

---

### Key Functional Requirements

1. **Processing Comparative Omics Data**:
   - Load tabular datasets from subfolders within the `primary_literature` directory.
   - Apply filters based on user-defined thresholds to identify **differentially expressed genes (DEGs)**.  
     - Define conditions for upregulated and downregulated genes.  
     - Ensure flexibility to handle various dataset formats (e.g., `.csv`, `.tsv`, `.xlsx`).
   - Support configuration-driven thresholds and parameters for comparative analysis.

2. **Object Persistence**:
   - Save results for each data source in the following formats:
     - **Plots**: `.pdf` plots visualizing DEGs (both up- and down-regulated).
     - **Data**: Resulting from efficient and decoupled subprocesses that generate data structures containing:
       - Identified DEGs (upregulated and downregulated).
       - Associated transcription factors (TFs).
       - Transcription factor binding sites (TFBSs) linked to the identified TFs.

3. **Decoupled Processes**:
   To ensure modularity and maintainability, implement the pipeline as three decoupled stages:
   1. **Differential Expression Analysis**:
      - Identify DEGs based on user-defined thresholds.
      - Return a tables of DEGs with relevant metadata.
   2. **Transcription Factor Association**:
      - Map identified DEGs to their associated transcription factors (TFs).
      - Use external or preloaded datasets for this mapping.
   3. **Transcription Factor Binding Site Identification**:
      - Identify TFBSs associated with the mapped TFs.
      - Implement this as a separate, reusable module or subprocess.

4. **Configurable Workflow**:
   - Use YAML configuration files to define:
     - Dataset-specific thresholds for DEG identification
     - Dataset-specific parameters (e.g., column names for reference/target conditions).
     - Output directories for `.pdf` plots and data.

5. **Intuitive and Descriptive Naming**:
   - Refactor module, class, and function names to clearly describe their purpose.
   - Consider renaming the subprocess/package to reflect its primary function (all lowercase, align with best practices in python naming).


### Implementation Considerations

1. **Modularity**:
   - Break the monolithic code into smaller, single-responsibility modules:
     - `deg_analysis.py` for DEG identification.
     - `tf_association.py` for TF mapping.
     - `tfbs_finder.py` for identifying TFBSs.

2. **Decoupling**:
   - Ensure each process (DEG analysis, TF mapping, TFBS identification) operates independently, with clear input/output contracts.

3. **Persistence**:
   - Save outputs (plots and data) in a dedicated directory structure that mirrors the input dataset hierarchy.

4. **Documentation**:
   - Provide clear, concise documentation and examples for using each module.

5. **Testing**:
   - Write unit tests for each module and integration tests for the overall workflow.

### Deliverables

1. **Restructured Codebase**:
   - A clean, modular directory structure for the subprocess.
   - If applicable, a new package name reflecting the functionality.

2. **Updated Documentation**:
   - Step-by-step guide for using the restructured pipeline.
   - Examples of input datasets and expected outputs.
