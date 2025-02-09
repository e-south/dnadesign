"""
--------------------------------------------------------------------------------
<deg2tfbs project>
utils.py

Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
from pathlib import Path
import pandas as pd

DNADESIGN_DATA = Path(__file__).parent.parent.parent.parent / 'dnadesign-data'

# A simple dictionary mapping keys to actual file paths:
DATA_FILES = {


    
    # Other Curated Literature
    'peebo': DNADESIGN_DATA / 'primary_literature' / 'Peebo_et_al' / 'gene_list.xlsx',

    # PRECISE-1K datasets from imodulonDB (gene sets following data-driven regulon categorization) (E. coli)
    'lamoureux_1k': DNADESIGN_DATA / 'primary_literature' / 'Lamoureux_et_al' / 'PRECISE_1K',
        
    # EcoCyc Pathway/Genome Database Full Regulatory Network
    'ecocyc_full_reg_network': DNADESIGN_DATA / 'EcoCyc' / 'ECOLI-regulatory-network.txt',
    'ecocyc_genes': DNADESIGN_DATA / 'RegulonDB_11' /'genes' / 'ecocyc_genes_and_synonyms.txt',
    
    # RegulonDB data sets
    'k_12_genome': DNADESIGN_DATA / 'RegulonDB_11' / 'K12_genome' / 'E_coli_K12_MG1655_U00096.3.txt',
    'regulondb_tf_set': DNADESIGN_DATA / 'RegulonDB_11' / 'tf_factors' / 'TFSet.csv',
    'regulondb_growth_condition_set': DNADESIGN_DATA / 'RegulonDB_11' /  'network_associations' / 'GCSet.txt',
    'regulondb_promoter_set': DNADESIGN_DATA / 'RegulonDB_11' / 'promoters' / 'PromoterSet.csv',
    'regulondb_promoter_predicted_set': DNADESIGN_DATA / 'RegulonDB_11' / 'promoters' / 'PromoterPredictionSet.csv',
    'regulondb_promoter_mendoza_race_set': DNADESIGN_DATA / 'RegulonDB_11' / 'promoters' / 'Promoter_from_RACE_Dataset.csv',
    'regulondb_promoter_mendoza_pyroseq_set': DNADESIGN_DATA / 'RegulonDB_11' / 'promoters' / 'Promoter_from_454_Dataset.csv',
    'regulondb_promoter_salgado_set': DNADESIGN_DATA / 'RegulonDB_11' / 'promoters' / 'ht_transcription_initiation_mapping_with_5_tri_or_monophosphate_enrichment_v3.0.csv',
    'regulondb_promoter_hernandez_set': DNADESIGN_DATA / 'primary_literature' / 'Hernandez et al' / 'positive2860.txt',
    'regulondb_non_promoter_hernandez_set': DNADESIGN_DATA / 'primary_literature' / 'Hernandez et al' / 'negative2860.txt',

    'regulondb_tfbs_prediction_medina_set': DNADESIGN_DATA / 'RegulonDB_11' / 'tf_binding_sites' / 'TFBSs_predictions_v3.txt',
    'regulondb_tfbs_prediction_hernandez_set': DNADESIGN_DATA / 'RegulonDB_11' / 'tf_binding_sites' / 'BindingSitePredictionSet.txt',
    'regulondb_tfbs_santos_set': DNADESIGN_DATA / 'RegulonDB_11' / 'tf_binding_sites' / 'BindingSiteSet.txt',
    'regulondb_tfbs_PSSM_set': DNADESIGN_DATA / 'RegulonDB_11' / 'tf_binding_sites' / 'RegulonDB_PSSM_v4.0' / 'results' / 'PSSM-Dataset-v4.0.txt',
    'regulondb_regulatory_interactions_set': DNADESIGN_DATA / 'RegulonDB_11' / 'network_associations' / 'RISet.txt',
    'regulondb_network_associations_tf_tf': DNADESIGN_DATA / 'RegulonDB_11' / 'network_associations' / 'network_tf_tf.txt',
    'regulondb_network_associations_tf_gene': DNADESIGN_DATA / 'RegulonDB_11' / 'network_associations' / 'network_tf_gene.txt',
    "target_set": DNADESIGN_DATA.parent.parent / 'decoydesigner' / 'data' / 'target_sets' / 'stress_and_growth_associated_tfs_20230603.csv',
    
    # High-throughput, functional promoter characterization datasets (E. coli)
    'thomason_tss_set': DNADESIGN_DATA / 'primary_literature' / 'Thomason_et_al' / 'zjb999093409sd1.xlsx',
    'johns_sequences': DNADESIGN_DATA / 'primary_literature' / 'Johns_et_al' / '41592_2018_BFnmeth4633_MOESM3_ESM.xlsx',
    'johns_expression_metrics': DNADESIGN_DATA / 'primary_literature' / 'Johns_et_al' / '41592_2018_BFnmeth4633_MOESM4_ESM.xlsx',
    'johns_expression_across_conditions': DNADESIGN_DATA / 'primary_literature' / 'Johns_et_al' / '41592_2018_BFnmeth4633_MOESM5_ESM.xlsx',
    'sun_yim': DNADESIGN_DATA / 'primary_literature' / 'Sun_Yim_et_al' / 'msb198875-sup-0002-sdatafig1.xlsx',
}


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
