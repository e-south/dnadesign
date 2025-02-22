"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/clustering/normalization.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import numpy as np


def normalize_array(arr, method="robust"):
    """
    Normalize an array using the specified method.
    
    - If method is 'z-score', compute (arr - mean) / std.
    - If method is 'robust' (the default), compute (arr - median) / IQR,
      where IQR is the interquartile range (75th percentile minus 25th percentile).
    """
    if method == "z-score":
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return arr - mean
        return (arr - mean) / std
    elif method == "robust":
        median = np.median(arr)
        q75, q25 = np.percentile(arr, 75), np.percentile(arr, 25)
        iqr = q75 - q25
        if iqr == 0:
            return arr - median
        return (arr - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def extract_numeric_hue(data_entries, numeric_hue_config, normalization_method="robust"):
    """
    Given the list of data_entries and a numeric_hue_config (a list of dictionaries
    mapping an input_source (subdirectory) to the meta key to use for numeric hue),
    first ensure that each source has exactly one meta key provided. If a source is
    assigned more than one meta key, raise an exception with a clear hint.

    Then, for each entry, extract the numeric value using the proper meta key
    (as determined by the entry's "meta_input_source").

    If there are entries from more than one source, apply normalization (with robust
    standardization by default) and print a clear message. If only one source is present,
    print that no normalization is needed.

    Returns:
        A NumPy array of numeric hue values (normalized if appropriate).
    """
    # Build mapping from source to meta key
    source_to_key = {}
    for item in numeric_hue_config:
        if not isinstance(item, dict):
            raise ValueError("Each numeric_hue config entry must be a dictionary mapping source to meta key.")
        if len(item) != 1:
            raise ValueError(f"Each numeric_hue config dictionary must have exactly one key-value pair, got: {item}.")
        for source, meta_key in item.items():
            if source in source_to_key:
                raise ValueError(f"Multiple hue keys provided for source '{source}'. Please ensure only one meta key is provided per source (or comment out the extras).")
            source_to_key[source] = meta_key

    # Extract values using the mapping
    hue_values = []
    for entry in data_entries:
        source = entry.get("meta_input_source")
        if source not in source_to_key:
            raise ValueError(f"Data entry source '{source}' not found in numeric_hue configuration. Please update your config.")
        meta_key = source_to_key[source]
        val = entry.get(meta_key, None)
        if val is None:
            raise ValueError(f"Data entry from source '{source}' is missing the expected meta key '{meta_key}'.")
        try:
            numeric_val = float(val)
        except Exception:
            raise ValueError(f"Value for key '{meta_key}' in source '{source}' is not numeric: {val}")
        hue_values.append(numeric_val)
    hue_values = np.array(hue_values)

    # Determine if we need to normalize
    unique_sources = set(entry.get("meta_input_source") for entry in data_entries)
    if len(unique_sources) > 1:
        print(f"Applying {normalization_method} normalization to numeric hue values across {len(unique_sources)} sources.")
        hue_values = normalize_array(hue_values, method=normalization_method)
    else:
        print("Only one numeric hue entry provided; skipping normalization.")

    return hue_values

def extract_type_hue(data_entries):
    """
    Extract hue values based on the part type. This function looks for the key
    'meta_part_type' in each data entry. If an entry is missing that key, an error is
    raised with a helpful hint.
    
    Returns:
        A list of type values (one per data entry).
    """
    type_values = []
    for entry in data_entries:
        if "meta_part_type" not in entry:
            raise ValueError("Data entry missing 'meta_part_type'. Please ensure every .pt file entry has this key (or update your configuration to use a different hue method).")
        type_values.append(entry["meta_part_type"])
    return type_values
