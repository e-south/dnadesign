"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/data_ingestor.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path
import pandas as pd
from dnadesign.utils import DEG2TFBS_DATA

class DEG2TFBSParser:
    def __init__(self, input_dir: str):
        p = Path(input_dir)
        if not p.is_absolute():
            self.source_dir = DEG2TFBS_DATA / p
        else:
            self.source_dir = p
        assert self.source_dir.exists() and self.source_dir.is_dir(), (
            f"Input directory {self.source_dir} does not exist or is not a directory."
        )

    def parse_tfbs_file(self, tfbsbatch_name: str) -> tuple[list[tuple[str, str, str]], pd.DataFrame]:
        """
        Parse the CSV file and return a list of (tf, tfbs, file_source) tuples,
        ensuring no duplicate pairs exist and neither tf nor tfbs is empty.
        Also return a trimmed metadata DataFrame with only selected meta_keys.
        """
        tfbsbatch_path = self.source_dir / tfbsbatch_name
        assert tfbsbatch_path.exists() and tfbsbatch_path.is_dir(), (
            f"Directory {tfbsbatch_path} does not exist."
        )
        csv_file = tfbsbatch_path / "tf2tfbs_mapping.csv"
        assert csv_file.exists(), f"CSV file {csv_file} does not exist."
        df = pd.read_csv(csv_file)
        required_columns = [
            'tf', 'tfbs', 'gene', 'deg_source', 
            'polarity', 'tfbs_source', 'is_sigma_factor', 
            'is_global_regulator'
        ]
        for col in required_columns:
            assert col in df.columns, f"Column {col} missing in {csv_file}."
        # Drop rows with missing or empty tf or tfbs, and drop duplicate (tf, tfbs) pairs.
        df = df.dropna(subset=['tf', 'tfbs'])
        df['tf'] = df['tf'].astype(str).str.strip()
        df['tfbs'] = df['tfbs'].astype(str).str.strip()
        df = df[df['tf'] != ""]
        df = df[df['tfbs'] != ""]
        df = df.drop_duplicates(subset=['tf', 'tfbs'])
        # Create a list of (tf, tfbs, file_source) tuples.
        file_source = csv_file.name  # e.g., "tf2tfbs_mapping.csv"
        pairs = [(row['tf'], row['tfbs'], file_source) for _, row in df.iterrows()]
        # Optionally, you might only want to keep a few meta keys.
        meta_df = df[['tf', 'tfbs', 'deg_source']]
        return pairs, meta_df

class DataIngestor:
    def __init__(self, input_dir: str, source_names: list):
        self.parser = DEG2TFBSParser(input_dir)
        self.source_names = source_names

    def load_all_data(self) -> tuple[list[tuple[str, str, str]], list[pd.DataFrame]]:
        """
        Loads data from all specified sources and returns:
         - a list of (tf, tfbs, file_source) tuples combined from all sources,
         - a list of DataFrames (one per source) for additional metadata.
        """
        all_pairs = []
        all_metadata = []
        for src in self.source_names:
            pairs, meta_df = self.parser.parse_tfbs_file(src)
            all_pairs.extend(pairs)
            all_metadata.append(meta_df)
        return all_pairs, all_metadata
