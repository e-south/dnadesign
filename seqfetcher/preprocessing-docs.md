## preprocessing
This subdirectory is part of the larger DNA **`dnadesign`** project and is responsible for processing data from primary literature and database sources. The **`preprocessing`** module organizes external data by wrangling sequences and their associated metadata into a list of Python dictionaries. These dictionaries are then standardized and saved as a `.pt` file using `torch.save(...)` for subsequent loading with `torch.load(...)`.

  ```python
  import torch

  # Toy example
  data = [
      {
          "id": "seq1",
          "sequence": "ACTG...",
          "meta_type": "promoter",
          "meta_source": "EcoCyc_Release_28_SmartTable_All_Promoters",
          # ...
      },
      {
          "id": "seq2",
          "sequence": "GATC...",
          "meta_type": "tfbs",
          "meta_source": "RegulonDB_Release_13_TFBSSet",
          # ...
      },
      # ...
  ]

  torch.save(data, "my_sequences.pt")
  ```

Then, to load:
```python
loaded_data = torch.load("my_sequences.pt")
```




## Subdirectory Structure

- **`data/`**  
  Contains datasets and resources for preprocessing. This directory has the following subdirectories:
  
  - **`primary_literature/`**  
    Contains datasets sourced from various RNA-seq studies, proteomic studies, and promoter engineering studies, primarily focused on *E. coli*. Detailed information about these datasets can be found in the `TOC.xlsx` file within this subdirectory. The `TOC.xlsx` file includes the following columns:
    - `folder_name`: Name of the subdirectory or dataset folder.
    - `title`: Title of the study or dataset.
    - `doi`: Digital Object Identifier (DOI) linking to the original publication.
    - `association`: Generic category flag.
    - `comments`: Additional notes or context about the dataset.

  - **`RegulonDB_11/`** and **`RegulonDB_13/`**  
    Contain experimental datasets from different releases of [RegulonDB](https://regulondb.ccg.unam.mx/). These directories store information relevant to gene regulation, operons, transcription factors, and other regulatory elements.  


  - **EcoCyc**

- **`src/`**  
  This directory contains Python modules that read, parse, and preprocess the datasets stored in `data/`. These modules are designed to:
  - Extract and clean data from literature-derived datasets.
  - Format data from RegulonDB releases for use in downstream analysis.
  - Harmonize diverse data formats into a unified structure for easier integration into **`sequencedesign`** workflows.

## Purpose

The `preprocessing` subdirectory is essential for ensuring that datasets from diverse sources are standardized and ready for further analysis in the DNA sequence design package. It handles tasks like:
- Reading experimental data from primary literature and RegulonDB releases.
- Cleaning and formatting raw datasets.
- Integrating metadata and annotations for downstream workflows.

---

**Author:** Eric J. South 
**Date:** January 19, 2025