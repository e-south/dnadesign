## seqfetcher

The **seqfetcher** subdirectory is designed to ingest and standardize DNA sequence data from the [**dnadesign-data**](https://github.com/e-south/dnadesign-data) repository. Each data source is represented by its own Python module that is responsible for:

1. **Loading Data:** Using the centralized `load_dataset()` helper from `utils.py` to locate and read files (Excel, CSV, TSV, or TXT) via relative paths defined in the `DATA_FILES` dictionary.

2. **Data Cleaning & Validation:** Each module cleans sequences (removing whitespace, ensuring uppercase A/T/C/G, etc.) and uses assertive checks to ensure that no entries have empty or invalid names or sequences.

3. **Standardizing Structure:** Data is transformed into a list of dictionaries. Each dictionary (representing a sequence) includes:
    - A unique identifier (`id`)
    - The DNA `sequence`
    - Metadata such as:
      - `meta_source` (e.g., the dataset key)
      - `meta_date_accessed`
      - `meta_part_type` (e.g., "promoter" or "tfbs")
      - Additional dataset-specific fields (e.g., transcription strength)

4. **Output Generation:** Each module saves the standardized data as a `.pt` file under a dedicated subfolder in the `sequences/` directory. A corresponding `summary.yaml` file is also generated to capture key details (source file, number of entries, timestamp, etc.).