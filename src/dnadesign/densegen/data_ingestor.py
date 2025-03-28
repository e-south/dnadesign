"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/data_ingestor.py

This module provides classes and functions to ingest tf2tfbs_mapping.csv data
from various sources. It supports:
  - Ingesting a CSV file from a deg2tfbs/pipeline/tfbsfetcher/ directory.
  - Ingesting CSV files from a deg2tfbs/analysis/outputs/ directory, where data is
    organized into cluster subdirectories.
  - Ingesting a PyTorch (.pt) file containing a list of dictionaries (each with a 
    key "sequence").

All input sources are defined under the "input_sources" key in the configuration,
and each source is processed independently.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import abc
from pathlib import Path
import pandas as pd
import torch

from dnadesign.utils import DEG2TFBS_DATA, BASE_DIR  # BASE_DIR is defined in utils.py

# --- Helper to resolve paths based on source type ---
def resolve_source_path(src_type: str, given_path: str) -> Path:
    """
    Resolves a relative path for deg2tfbs sources.
    For type "deg2tfbs_pipeline_tfbsfetcher": BASE_DIR / 'src/deg2tfbs/pipeline/tfbsfetcher' / given_path
    For type "deg2tfbs_cluster_analysis": BASE_DIR / 'src/deg2tfbs/analysis/outputs' / given_path
    For type "pt": BASE_DIR / 'src/dnadesign/sequences' / given_path
    If given_path is absolute, it is returned as is.
    """
    p = Path(given_path)
    if p.is_absolute():
        return p
    if src_type.lower() == "deg2tfbs_pipeline_tfbsfetcher":
        return DEG2TFBS_DATA / "src" / "deg2tfbs" / "pipeline" / "tfbsfetcher" / p
    elif src_type.lower() == "deg2tfbs_cluster_analysis":
        return DEG2TFBS_DATA / "src" / "deg2tfbs" / "analysis" / "outputs" / p
    elif src_type.lower() == "pt":
        return BASE_DIR / "src" / "dnadesign" / "sequences" / p
    else:
        raise ValueError(f"Unsupported source type: {src_type}")

# --- Abstract Base Class ---
class BaseDataSource(abc.ABC):
    @abc.abstractmethod
    def load_data(self):
        """
        Loads data from the source.

        Preconditions:
          - Any required keys and files exist.
        Postconditions:
          - Returns a tuple (data_entries, meta_data) where:
              * data_entries: a list of data entries (e.g., tuples or dictionaries).
              * meta_data: additional metadata (e.g., a pandas DataFrame) or None.
        """
        pass

# --- CSV File Data Source (for tfbsfetcher) ---
class CSVDataSource(BaseDataSource):
    def __init__(self, directory: str, src_type: str = "deg2tfbs_pipeline_tfbsfetcher"):
        """
        Initializes a CSVDataSource for a single tfbsbatch directory.
        
        Preconditions:
          - directory is a valid directory path (relative paths are resolved appropriately)
            containing a file named "tf2tfbs_mapping.csv".
        """
        self.dir = resolve_source_path(src_type, directory)
        assert self.dir.exists() and self.dir.is_dir(), f"Directory {self.dir} must exist and be a directory."
    
    def load_data(self):
        csv_file = self.dir / "csvs" / "tf2tfbs_mapping.csv"
        assert csv_file.exists(), f"CSV file {csv_file} does not exist."
        df = pd.read_csv(csv_file)
        required_columns = [
            'tf', 'tfbs', 'gene', 'deg_source', 
            'polarity', 'tfbs_source', 'is_sigma_factor', 
            'is_global_regulator'
        ]
        for col in required_columns:
            assert col in df.columns, f"Required column {col} missing in {csv_file}."
        df = df.dropna(subset=['tf', 'tfbs'])
        df['tf'] = df['tf'].astype(str).str.strip()
        df['tfbs'] = df['tfbs'].astype(str).str.strip()
        df = df[df['tf'] != ""]
        df = df[df['tfbs'] != ""]
        df = df.drop_duplicates(subset=['tf', 'tfbs'])
        data_entries = list(zip(df['tf'].tolist(), df['tfbs'].tolist(), [str(csv_file)] * len(df)))
        return data_entries, df

# --- Cluster Data Source (for analysis outputs) ---
class ClusterDataSource(BaseDataSource):
    def __init__(self, parent_dir: str, clusters: list, src_type: str = "deg2tfbs_cluster_analysis"):
        self.parent_dir = resolve_source_path(src_type, parent_dir)
        assert self.parent_dir.exists() and self.parent_dir.is_dir(), (
            f"Parent directory {self.parent_dir} does not exist or is not a directory."
        )
        assert clusters and isinstance(clusters, list), "clusters must be a non-empty list."
        self.clusters = clusters
    
    def load_data(self):
        all_entries = []
        meta_frames = []
        for cluster in self.clusters:
            cluster_dir = self.parent_dir / cluster
            assert cluster_dir.exists() and cluster_dir.is_dir(), f"Cluster directory {cluster_dir} does not exist."
            csv_file = cluster_dir / "tf2tfbs_mapping.csv"
            assert csv_file.exists(), f"CSV file {csv_file} does not exist in cluster {cluster}."
            df = pd.read_csv(csv_file)
            required_columns = [
                'tf', 'tfbs', 'gene', 'deg_source', 
                'polarity', 'tfbs_source', 'is_sigma_factor', 
                'is_global_regulator'
            ]
            for col in required_columns:
                assert col in df.columns, f"Required column {col} missing in {csv_file}."
            df = df.dropna(subset=['tf', 'tfbs'])
            df['tf'] = df['tf'].astype(str).str.strip()
            df['tfbs'] = df['tfbs'].astype(str).str.strip()
            df = df[df['tf'] != ""]
            df = df[df['tfbs'] != ""]
            df = df.drop_duplicates(subset=['tf', 'tfbs'])
            entries = list(zip(df['tf'].tolist(), df['tfbs'].tolist(), [str(cluster_dir)] * len(df)))
            all_entries.extend(entries)
            meta_frames.append(df)
        combined_meta = pd.concat(meta_frames, ignore_index=True) if meta_frames else None
        return all_entries, combined_meta

# --- PyTorch File Data Source ---
class PTDataSource(BaseDataSource):
    def __init__(self, pt_file: str):
        self.pt_file = resolve_source_path("pt", pt_file)
        assert self.pt_file.exists() and self.pt_file.is_file(), f"PT file {self.pt_file} does not exist."
    
    def load_data(self):
        data = torch.load(self.pt_file)
        for entry in data:
            assert "sequence" in entry, f"An entry in {self.pt_file} is missing the 'sequence' key."
        return data, None

# --- Factory Function ---
def data_source_factory(source_config: dict) -> BaseDataSource:
    assert "type" in source_config, "source_config must include a 'type' key."
    assert "path" in source_config, "source_config must include a 'path' key."
    src_type = source_config["type"].lower()
    path = source_config["path"]
    if src_type == "deg2tfbs_pipeline_tfbsfetcher":
        return CSVDataSource(path, src_type="deg2tfbs_pipeline_tfbsfetcher")
    elif src_type == "deg2tfbs_cluster_analysis":
        assert "clusters" in source_config, "For analysis type, 'clusters' key must be provided."
        clusters = source_config["clusters"]
        return ClusterDataSource(path, clusters, src_type="deg2tfbs_cluster_analysis")
    elif src_type == "pt":
        return PTDataSource(path)
    else:
        raise ValueError(f"Unsupported source type: {src_type}")

# --- Main Ingestor Class ---
class TFBSMappingDataIngestor:
    def __init__(self, source_configs: list):
        assert source_configs and isinstance(source_configs, list), "source_configs must be a non-empty list."
        self.sources = [data_source_factory(cfg) for cfg in source_configs]
    
    def load_all_sources(self) -> tuple[list, list]:
        all_entries = []
        all_meta = []
        for src in self.sources:
            entries, meta = src.load_data()
            all_entries.extend(entries)
            all_meta.append(meta)
        return all_entries, all_meta
