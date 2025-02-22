"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/sequences/seqmanager.py

This module performs quality control and testing of (.pt) files.
When run as __main__, it accepts a --path argument that may be either a file or
a directory. If a file is provided (along with an --index, and optionally a --key),
the module will load that file (assumed to reside in the local directory or subdir)
and print the requested entry (or the entire entry if no key is provided).
If a directory is provided (or no --path is provided, defaulting to the current dir),
the module recursively scans for .pt files, validates each one, and prints feedback,
including a summary of how many files passed.

Usage examples:
  # Validate all .pt files under the current directory:
  $ python seqmanager.py --path .
  
  # Validate all .pt files in a specific directory:
  $ python seqmanager.py --path seqbatch_random_tfbs
  
  # Inspect entry 3 from a file:
  $ python seqmanager.py --path densebatch_deg2tfbs_pipeline_tfbsfetcher_all_DEG_sets_n2500/densegenbatch_all_DEG_sets_n2500.pt --index 3
  $ python seqmanager.py --path densebatch_deg2tfbs_cluster_analysis_unfiltered_cluster_3_n2500/densegenbatch_unfiltered_cluster_3_n2500.pt --index 3
  $ python seqmanager.py --path densebatch_pt_seqbatch_random_tfbs_n2500/densegenbatch_seqbatch_random_tfbs_n2500.pt --index 3
  $ python seqmanager.py --path seqbatch_hossain_et_al/seqbatch_hossain_et_al.pt --index 3
  $ python seqmanager.py --path seqbatch_johns_et_al/seqbatch_johns_et_al.pt --index 3
  $ python seqmanager.py --path seqbatch_sun_yim_et_al/seqbatch_sun_yim_et_al.pt --index 3
  
  # Inspect only the 'sequence' key of entry 3:
  $ python seqmanager.py --path seqbatch_random_tfbs/seqbatch_random_tfbs.pt --index 3 --key sequence

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import argparse
from pathlib import Path
import torch
import yaml

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def validate_pt_file(file_path: str) -> bool:
    """
    Loads a .pt file, verifies that it is a non-empty list of dictionaries, and
    that each dictionary contains the "sequence" key.
    """
    pt_path = Path(file_path)
    assert pt_path.exists() and pt_path.is_file(), f"PT file {pt_path} does not exist."
    checkpoint = torch.load(pt_path, map_location=torch.device('cpu'))
    assert isinstance(checkpoint, list) and len(checkpoint) > 0, f"{pt_path} must be a non-empty list."
    for i, entry in enumerate(checkpoint):
        assert isinstance(entry, dict), f"Entry {i} in {pt_path} is not a dictionary."
        assert "sequence" in entry, f"Entry {i} in {pt_path} is missing the 'sequence' key."
    return True

def inspect_entry(file_path: str, index: int) -> dict:
    """
    Loads a .pt file and returns the entry (a dictionary) at the given index.
    """
    pt_path = Path(file_path)
    checkpoint = torch.load(pt_path, map_location=torch.device('cpu'))
    if not (0 <= index < len(checkpoint)):
        raise IndexError(f"Index {index} is out of range for file {pt_path} (length {len(checkpoint)}).")
    return checkpoint[index]

def list_all_pt_files(search_path: str) -> list:
    """
    Recursively searches the given directory for all .pt files.
    Returns a list of Path objects.
    """
    base_path = Path(search_path)
    assert base_path.exists() and base_path.is_dir(), f"Directory {base_path} does not exist."
    return list(base_path.rglob("*.pt"))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and inspect .pt files for densegen."
    )
    parser.add_argument(
        "--path", type=str, default=".",
        help="File or directory path. If a file is given, --index (and optionally --key) must be provided."
    )
    parser.add_argument(
        "--index", type=int,
        help="(Optional) Index of the entry to inspect (if --path is a file)."
    )
    parser.add_argument(
        "--key", type=str,
        help="(Optional) Specific key from the entry to display."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    target_path = Path(args.path)
    # If the target path is a file, then operate in inspection mode.
    if target_path.is_file():
        # Validate the file and then inspect.
        try:
            validate_pt_file(str(target_path))
            print(f"File '{target_path}' is VALID.")
        except AssertionError as e:
            print(f"Validation failed for '{target_path}': {e}")
            return
        if args.index is None:
            print("Error: When a file is specified, you must provide --index to inspect an entry.")
            return
        try:
            entry = inspect_entry(str(target_path), args.index)
            print(f"\nEntry at index {args.index} in '{target_path}':")
            if args.key:
                if args.key in entry:
                    print(f"{args.key}: {entry[args.key]}")
                else:
                    print(f"Key '{args.key}' not found in entry.")
            else:
                # Pretty-print the entire entry.
                for k, v in entry.items():
                    print(f"  {k}: {v}")
        except Exception as e:
            print(f"Error inspecting entry: {e}")
    else:
        # Assume target_path is a directory.
        pt_files = list_all_pt_files(str(target_path))
        if not pt_files:
            print(f"No .pt files found in directory '{target_path}'.")
            return
        valid_files = []
        print("Validating .pt files:")
        for pt_file in pt_files:
            try:
                if validate_pt_file(str(pt_file)):
                    print(f"  VALID: {pt_file}")
                    valid_files.append(pt_file)
            except AssertionError as e:
                print(f"  INVALID: {pt_file} --> {e}")
        print(f"\nSummary: {len(valid_files)} out of {len(pt_files)} .pt files in '{target_path}' are valid.")

if __name__ == "__main__":
    main()
