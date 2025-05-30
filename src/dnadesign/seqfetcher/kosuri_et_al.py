"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/kosuri_et_al.py

Module for loading data described in Kosuri et al., which synthesized 12,563 
combinations of common promoters and ribosome binding sites and simultaneously 
measured DNA, RNA, and protein levels from the entire library.

"Composability of regulatory sequences controlling transcription and translation 
in Escherichia coli"
DOI: 10.1073/pnas.1301301110

Ingests the Kosuri et al promoter dataset from:
    DATA_FILES["kosuri_et_al"]
Sheet: "Promoters"
Columns:
  - Name: "Promoter"
  - Sequence: "Sequence" (which contains an initial cut site and a barcode)
    → The cleaning function removes the cut site (first token) and drops the last five nucleotides.
  - Additional meta: "mean.RNA", "sd.RNA", "mean.prot", "sd.prot"
Sets meta_part_type to "promoter".

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import sys
from pathlib import Path

# Add the src directory to sys.path so that "dnadesign" can be found.
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent.parent
sys.path.insert(0, str(src_dir))

import datetime
import re
import uuid

import pandas as pd

from dnadesign.utils import BASE_DIR, SequenceSaver, load_dataset

VALID_NUCLEOTIDES = set("ATCG")


def clean_sequence(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    seq = seq.strip()
    # Expect two parts: [cut_site, promoter_sequence_with_barcode]
    parts = seq.split()
    if len(parts) < 2:
        return ""
    promoter_seq = parts[1]
    # Remove last five nucleotides (barcode)
    if len(promoter_seq) > 5:
        promoter_seq = promoter_seq[:-5]
    promoter_seq = promoter_seq.upper()
    promoter_seq = re.sub(r"\s+", "", promoter_seq)
    return "".join(c for c in promoter_seq if c in VALID_NUCLEOTIDES)


def validate_entry(name: str, seq: str):
    if not name or pd.isna(name):
        raise AssertionError("Empty name field.")
    if not seq:
        raise AssertionError("Empty sequence field.")
    for c in seq:
        if c not in VALID_NUCLEOTIDES:
            raise AssertionError(f"Invalid nucleotide '{c}' in sequence: {seq}")


def ingest():
    df = load_dataset("kosuri_et_al", sheet_name="Promoters")
    sequences = []
    for idx, row in df.iterrows():
        name = row.get("Promoter")
        # Remove any surrounding quotation marks and whitespace from the name.
        if isinstance(name, str):
            name = name.strip().strip('"')
        seq_raw = row.get("Sequence")
        mean_rna = row.get("mean.RNA")
        sd_rna = row.get("sd.RNA")
        mean_prot = row.get("mean.prot")
        sd_prot = row.get("sd.prot")
        if pd.isna(name):
            name = f"Row_{idx}"
        if pd.isna(seq_raw):
            continue
        seq = clean_sequence(seq_raw)
        try:
            validate_entry(name, seq)
        except AssertionError as e:
            print(f"Skipping row {idx}: {e}")
            continue
        entry = {
            "id": str(uuid.uuid4()),
            "name": name,
            "sequence": seq,
            "meta_source": "kosuri_et_al",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_mean_RNA": mean_rna,
            "meta_sd_RNA": sd_rna,
            "meta_mean_prot": mean_prot,
            "meta_sd_prot": sd_prot,
            "meta_part_type": "engineered promoter",
        }
        sequences.append(entry)
    return sequences


def save_output(sequences):
    output_dir = Path(BASE_DIR) / "src" / "dnadesign" / "sequences" / "seqbatch_kosuri_et_al"
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    additional_info = {"source_file": "kosuri_et_al", "part_type": "promoter"}
    saver.save_with_summary(sequences, "seqbatch_kosuri_et_al.pt", additional_info=additional_info)


if __name__ == "__main__":
    seqs = ingest()
    save_output(seqs)
