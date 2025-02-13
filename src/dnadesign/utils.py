"""
--------------------------------------------------------------------------------
<dnadesign project>
utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from pathlib import Path
import torch
import pandas as pd
import yaml
import datetime
import uuid


BASE_DIR = Path(__file__).resolve().parent[2]
DEG2TFBS_DATA = BASE_DIR.parent / "deg2tfbs"
DNADESIGN_DATA = BASE_DIR.parent / 'dnadesign-data'

# A dictionary mapping keys to actual file paths:
DATA_FILES = {
    # Promoter Engineering Datasets
    "hossain_et_al": DNADESIGN_DATA / 'primary_literature' / 'LaFleur_et_al' / '41467_2022_32829_MOESM5_ESM.xlsx',
    "kosuri_et_al": DNADESIGN_DATA / 'primary_literature' / 'LaFleur_et_al' / 'sd01.xls',
    "lafleur_et_al": DNADESIGN_DATA / 'primary_literature' / 'LaFleur_et_al' / '41467_2022_32829_MOESM5_ESM.xlsx',
    "urtecho_et_al": DNADESIGN_DATA / 'primary_literature' / 'LaFleur_et_al' / '41467_2022_32829_MOESM5_ESM.xlsx',
    "yu_et_al": DNADESIGN_DATA / 'primary_literature' / 'LaFleur_et_al' / '41467_2022_32829_MOESM5_ESM.xlsx',
    
    # EcoCyc Promoter SmartTable(s)
    'ecocyc_28_promoters': DNADESIGN_DATA / 'EcoCyc_28' / 'SmartTable_All_Promoters.txt',
    
    # RegulonDB Promoter Datasets
    'regulondb_13_promoter_set': DNADESIGN_DATA / 'RegulonDB_13' / 'promoters' / 'PromoterSet.tsv',
    'regulondb_13_promoter_FecI_set': DNADESIGN_DATA / 'RegulonDB_13' / 'promoters' / 'FecI_RDBECOLISFC00002_SigmulonPromoters.csv',    
    'regulondb_13_promoter_FliA_set': DNADESIGN_DATA / 'RegulonDB_13' / 'promoters' / 'FliA_RDBECOLISFC00001_SigmulonPromoters.csv',
    'regulondb_13_promoter_RpoD_set': DNADESIGN_DATA / 'RegulonDB_13' / 'promoters' / 'RpoD_RDBECOLISFC00003_SigmulonPromoters.csv',
    'regulondb_13_promoter_RpoE_set': DNADESIGN_DATA / 'RegulonDB_13' / 'promoters' / 'RpoE_RDBECOLISFC00004_SigmulonPromoters.csv',
    'regulondb_13_promoter_RpoH_set': DNADESIGN_DATA / 'RegulonDB_13' / 'promoters' / 'RpoH_RDBECOLISFC00005_SigmulonPromoters.csv',
    'regulondb_13_promoter_RpoN_set': DNADESIGN_DATA / 'RegulonDB_13' / 'promoters' / 'RpoN_RDBECOLISFC00006_SigmulonPromoters.csv',
    'regulondb_13_promoter_RpoS_set': DNADESIGN_DATA / 'RegulonDB_13' / 'promoters' / 'RpoS_RDBECOLISFC00007_SigmulonPromoters.csv',
    
    # EcoCyc Transcription Factor Binding Site SmartTable(s)
    'ecocyc_28_tf_binding_sites': DNADESIGN_DATA / 'EcoCyc_28' / 'SmartTable_All_Transcription_Factor_Binding_Sites.txt',
        
    # RegulonDB Transcription Factor Binding Site Dataset(s)
    'regulondb_13_tf_ri_set': DNADESIGN_DATA / 'RegulonDB_13' / 'binding_sites' / 'TF-RISet.tsv',

    # Other Datasets
    'k_12_genome': DNADESIGN_DATA / 'RegulonDB_11' / 'K12_genome' / 'E_coli_K12_MG1655_U00096.3.txt',
    'regulondb_tf_set': DNADESIGN_DATA / 'RegulonDB_11' / 'tf_factors' / 'TFSet.csv',
}

class ConfigLoader:
    def __init__(self, config_path: Path):
        assert config_path.exists(), f"Config file {config_path} does not exist."
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        with self.config_path.open("r") as f:
            config = yaml.safe_load(f)
        return config.get("densegen", {})

class SequenceSaver:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, sequences: list, filename: str):
        file_path = self.output_dir / filename
        torch.save(sequences, file_path)
        print(f"Saved {len(sequences)} entries to {file_path}")


def generate_sequence_entry(dense_array_solution, source_names: list, tfbs_parts: list, config: dict) -> dict:
    """
    Generate a flat dictionary representing the dense array solution with metadata.
    :param dense_array_solution: The solution returned from the optimizer.
    :param source_names: List of source names.
    :param tfbs_parts: List of strings for each TF–TFBS pairing.
    :param config: Dictionary of configuration parameters.
    """
    seq_id = str(uuid.uuid4())
    date_accessed = datetime.datetime.now().isoformat()
    meta_source = "deg2tfbs_" + "_AND_".join(source_names)
    entry = {
        "id": seq_id,
        "meta_date_accessed": date_accessed,
        "meta_source": meta_source,
        "meta_sequence_visual": str(dense_array_solution),
        "meta_offsets": (dense_array_solution.offset_indices_in_order()
                         if hasattr(dense_array_solution, "offset_indices_in_order") else None),
        "meta_compression_ratio": getattr(dense_array_solution, "compression_ratio", None),
        "meta_nb_motifs": getattr(dense_array_solution, "nb_motifs", None),
        "meta_gap_fill": getattr(dense_array_solution, "meta_gap_fill", False),
        "meta_gap_fill_details": getattr(dense_array_solution, "meta_gap_fill_details", None),
        "meta_tfbs_parts": tfbs_parts,
        "tfs_sample": source_names,
        "sequence": dense_array_solution.sequence,
        "config": {
            "sequence_length": config.get("sequence_length"),
            "quota": config.get("quota"),
            "subsample_size": config.get("subsample_size"),
            "arrays_generated_before_resample": config.get("arrays_generated_before_resample"),
            "solver": config.get("solver"),
            "solver_options": config.get("solver_options"),
            "fixed_elements": config.get("fixed_elements")
        }
    }
    return entry


def load_dataset(dataset_key, sheet_name=None, usecols=None, header=0, skiprows=None):
    """
    Minimal example that loads a dataset from the known path in DATA_FILES.
    If it's an Excel, we read via pd.read_excel; if CSV, pd.read_csv, etc.
    """
    file_path = DATA_FILES[dataset_key]
    if not file_path.exists():
        raise FileNotFoundError(f"File not found for key {dataset_key}: {file_path}")

    # Fix: Treat both .xlsx and .xls as Excel
    if file_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(
            file_path, 
            sheet_name=sheet_name,
            usecols=usecols,
            header=header,
            skiprows=skiprows
        )
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, usecols=usecols, header=header)
    else:
        raise ValueError(f"Unsupported file format for {file_path.suffix}")

    return df