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


BASE_DIR = Path(__file__).resolve().parent
DEG2TFBS_DATA = BASE_DIR.parent / "deg2tfbs"
DNADESIGN_DATA = BASE_DIR.parent / 'dnadesign-data'

# A dictionary mapping keys to actual file paths:
DATA_FILES = {
    # EcoCyc Pathway/Genome Database Full Regulatory Network File(s)
    'ecocyc_28_reg_network': DNADESIGN_DATA / 'EcoCyc_28' / 'ECOLI-regulatory-network.txt',
    
    # RegulonDB Regulatory Network File(s)
    'regulondb_13_network_interactions': DNADESIGN_DATA / 'RegulonDB_13' / 'network_interactions' / 'NetworkRegulatorGene.tsv',
    
    # EcoCyc Transcription Factor Binding Site File(s)
    'ecocyc_28_tfbs_smart_table': DNADESIGN_DATA / 'EcoCyc_28' / 'SmartTable_All_Transcription_Factor_Binding_Sites.txt',
    
    # RegulonDB Transcription Factor Binding Site File(s)
    'regulondb_13_tf_ri_set': DNADESIGN_DATA / 'RegulonDB_13' / 'binding_sites' / 'TF-RISet.tsv',

    # Other Datasets
    'ecocyc_genes': DNADESIGN_DATA / 'RegulonDB_11' /'genes' / 'ecocyc_genes_and_synonyms.txt',
    'k_12_genome': DNADESIGN_DATA / 'RegulonDB_11' / 'K12_genome' / 'E_coli_K12_MG1655_U00096.3.txt',
    'regulondb_tf_set': DNADESIGN_DATA / 'RegulonDB_11' / 'tf_factors' / 'TFSet.csv',
    'regulondb_growth_condition_set': DNADESIGN_DATA / 'RegulonDB_11' /  'network_associations' / 'GCSet.txt',
    'regulondb_tfbs_prediction_medina_set': DNADESIGN_DATA / 'RegulonDB_11' / 'tf_binding_sites' / 'TFBSs_predictions_v3.txt',
    
    # Promoter Engineering Datasets
    "hossain_et_al": None,
    "kosuri_et_al": None,
    "lafleur_et_al": None,
    "urtecho_et_al": None,
    "yo_et_al": None,
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

def generate_sequence_entry(dense_array_solution, source_names: list, tfbs_parts: list) -> dict:
    """
    Generate a flat dictionary representing the dense array solution with metadata.
    :param dense_array_solution: The solution returned from the optimizer.
    :param source_names: List of source names.
    :param tfbs_parts: List of strings for each TF-TFBS pairing.
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
        "sequence": dense_array_solution.sequence
    }
    return entry