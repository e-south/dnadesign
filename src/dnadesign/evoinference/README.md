## evoinference

**evoinference** is a pipeline that leverages the [Evo 2](https://github.com/ArcInstitute/evo2) model for genomic inference. It processes batches of `.pt` files (located in the sibling **sequences** directory) by performing the following steps for each sequence entry:

- **Tokenization & Model Inference:** Converts sequences into tensors, runs a forward pass through Evo 2, and extracts outputs—either final-layer or intermediate-layer embeddings.
- **Output Augmentation:** Optionally applies mean pooling across the sequence length (e.g., reducing an n × 512 tensor to a 1 × 512 vector) and adds tensor shape metadata.
- **Result Storage:** Updates the original `.pt` files with new keys containing the model outputs while logging progress and handling checkpoints.


### Quick Start

For full setup instructions—including environment configuration, CUDA module loading, and Evo 2 installation—refer to the **Installation Instructions** in the main project [**README**](../../../README.md#installation).

1. **Configure Your Run:**  
   Edit the YAML configuration file (e.g., `../configs/example.yaml`) to specify:
   - **Data sources:** Subdirectories under **sequences** to be processed.
   - **Model settings:** Choose the Evo 2 model version (default is `evo2_7b`), and set output types and pooling options.
  
2. **Run the Pipeline:**  
   From the **evoinference** directory, launch the pipeline:
   ```bash
   python main.py
   ```
   The script automatically loads sequences, processes each file, updates results, and writes checkpoints.


### Module Overview

- **main.py:**  
  The entry script which loads the configuration, initializes the Evo 2 model, and processes each `.pt` file.
  
- **config.py:**  
  Loads and validates the YAML configuration.
  
- **data_ingestion.py:**  
  Lists, loads, and validates input `.pt` files containing sequence data.
  
- **model_invocation.py:**  
  Initializes Evo 2, tokenizes the sequences, and runs the inference.
  
- **storage.py:**  
  Handles writing updated results and checkpoints to files.
  
- **aggregations.py:**  
  Augments raw model outputs with pooling and shape metadata.
  
- **logger.py:**  
  Provides logging throughout the module.
