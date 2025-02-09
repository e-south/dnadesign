import pandas as pd
from pathlib import Path

def load_tsv(file_path, header_line):
    """Load a TSV file, skipping lines until the header."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, sep='\t', skiprows=header_line - 1)

def clean_tfrs_seq(data, seq_column):
    """Strip lowercase characters from the left and right of the sequence column."""
    if seq_column not in data.columns:
        raise ValueError(f"Column '{seq_column}' not found in data.")
    data[seq_column] = data[seq_column].str.lstrip('abcdefghijklmnopqrstuvwxyz').str.rstrip('abcdefghijklmnopqrstuvwxyz')
    return data

def format_as_csv(data, name_column, seq_column, output_path):
    """Format the data as a CSV with no quotes."""
    if name_column not in data.columns or seq_column not in data.columns:
        raise ValueError(f"Required columns '{name_column}' and '{seq_column}' not found in data.")
    data = data[[name_column, seq_column]]
    data = data.dropna(subset=[seq_column])
    data.to_csv(output_path, index=False, header=False, quoting=3)

def process_tsv_to_csv(input_file, output_file, header_line, name_column, seq_column):
    """Main function to process the TSV and save the cleaned CSV."""
    data = load_tsv(input_file, header_line)
    data = clean_tfrs_seq(data, seq_column)
    format_as_csv(data, name_column, seq_column, output_file)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "RegulonDB_13"
    INPUT_FILE = BASE_DIR / "TF-RISet.tsv"
    OUTPUT_FILE = BASE_DIR / "Tidy_Regulators_TFBS_Sequences.csv"
    HEADER_LINE = 45
    NAME_COLUMN = "4)regulatorName"
    SEQ_COLUMN = "10)tfrsSeq"

    try:
        process_tsv_to_csv(INPUT_FILE, OUTPUT_FILE, HEADER_LINE, NAME_COLUMN, SEQ_COLUMN)
        print(f"Processed file saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error: {e}")
