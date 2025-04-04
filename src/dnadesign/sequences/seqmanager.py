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
including a summary of how many files passed. When provided  --update-meta, the module
updates (or adds) the 'meta_part_type' key for each entry in the .pt file(s) based 
on the parent directory. If the first entry already has the key, the user is prompted
to either skip or overwrite (update all entries).

Usage examples:
  # Validate all .pt files under the current directory:
  $ python seqmanager.py --path .
  
  # Validate all .pt files in a specific directory:
  $ python seqmanager.py --path seqbatch_random_tfbs
  
  # Inspect entry 3 from a file:
  $ python seqmanager.py --path seqbatch_hossain_et_al/seqbatch_hossain_et_al.pt --index 3
  $ python seqmanager.py --path seqbatch_johns_et_al/seqbatch_johns_et_al.pt --index 3
  $ python seqmanager.py --path seqbatch_sun_yim_et_al/seqbatch_sun_yim_et_al.pt --index 3
  $ python seqmanager.py --path densebatch_test/densegenbatch_m9_acetate_tfs_n10000.pt --index 3
  $ python seqmanager.py --path densebatch_deg2tfbs_pipeline_tfbsfetcher_m9_acetate_tfs_n10000_subsample20_diverse/densegenbatch_m9_acetate_tfs_n10000.pt --index 3
  $ python seqmanager.py --path seqbatch_random_promoters_sigma70_consensus/seqbatch_random_promoters_sigma70_consensus.pt --index 3
  $ python seqmanager.py --path seqbatch_xiaowo_et_al/seqbatch_xiaowo_et_al.pt --index 3
  
  # Inspect only the 'sequence' key of entry 3:
  $ python seqmanager.py --path seqbatch_random_tfbs/seqbatch_random_tfbs.pt --index 3 --key sequence
  
  # Update the meta_part_type key for a single file:
  $ python seqmanager.py --path seqbatch_hossain_et_al/seqbatch_hossain_et_al.pt --update-meta
  
  # Recursively update meta_part_type in all .pt files under a directory:
  $ python seqmanager.py --path . --update-meta

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import argparse
from pathlib import Path
import torch
import yaml
import warnings
import re
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

def update_meta_part_type(file_path: str) -> None:
    """
    Loads a .pt file and for each entry ensures that the 'meta_part_type' key exists.
    The desired value is determined based on the parent directory's name using a lookup
    (including handling a wildcard for 'densebatch_*'). For directories starting with
    'densebatch_', the user is prompted to decide whether to define dense arrays by constraint.
    
    If the user chooses yes, then the file's name is examined via regex to assign:
      - 'sigma70' if 'sigma70' is in the file name,
      - 'sigma24' if 'sigma24' is in the file name,
      - 'sigma38' if 'sigma38' is in the file name,
      - 'sigma32' if 'sigma32' is in the file name.
    Otherwise (or if no match is found), the file is labeled as a dense array.
    
    If the first entry already has the key, the user is prompted to either skip or overwrite.
    The file is then saved in-place.
    """
    pt_path = Path(file_path)
    checkpoint = torch.load(pt_path, map_location=torch.device('cpu'))
    if not (isinstance(checkpoint, list) and len(checkpoint) > 0):
        print(f"File {pt_path} is not a valid .pt file format for meta update.")
        return

    # Determine the desired meta_part_type based on the parent directory name.
    parent_dir = pt_path.parent.name
    meta_mapping = {
        "seqbatch_hossain_et_al": "engineered promoter",
        "seqbatch_kosuri_et_al": "engineered promoter",
        "seqbatch_lafleur_et_al": "engineered promoter",
        "seqbatch_sun_yim_et_al": "engineered promoter",
        "seqbatch_urtecho_et_al": "engineered promoter",
        "seqbatch_yu_et_al": "engineered promoter",
        "seqbatch_johns_et_al": "engineered promoter",
        "seqbatch_hernandez_et_al_negative": "natural non-promoter",
        "seqbatch_hernandez_et_al_positive": "natural promoter",
        "seqbatch_random_promoters": "random promoter",
        "seqbatch_random_tfbs": "random TFBS",
        "seqbatch_regulondb_13_promoter_FecI_set": "natural promoter",
        "seqbatch_regulondb_13_promoter_FliA_set": "natural promoter",
        "seqbatch_regulondb_13_promoter_RpoD_set": "natural promoter",
        "seqbatch_regulondb_13_promoter_RpoE_set": "natural promoter",
        "seqbatch_regulondb_13_promoter_RpoH_set": "natural promoter",
        "seqbatch_regulondb_13_promoter_RpoN_set": "natural promoter",
        "seqbatch_regulondb_13_promoter_RpoS_set": "natural promoter",
        "seqbatch_regulondb_13_promoter_set": "natural promoter",
        "seqbatch_regulondb_13_tf_ri_set": "natural TFBS",
        "seqbatch_ecocyc_28_promoters": "natural promoter",
        "seqbatch_ecocyc_28_tfbs_set": "natural TFBS"
    }
    
    if parent_dir.startswith("densebatch_"):
        # Ask the user if they want to define dense arrays by constraint.
        user_input = input(
            f"Directory '{parent_dir}' is a densebatch directory. Would you like to define dense arrays by constraint? [y/n]: "
        ).strip().lower()
        if user_input.startswith("y"):
            # Apply regex (or simple substring search) on the file name.
            file_stem = pt_path.stem.lower()
            if "sigma70" in file_stem:
                desired_value = "sigma70"
            elif "sigma24" in file_stem:
                desired_value = "sigma24"
            elif "sigma38" in file_stem:
                desired_value = "sigma38"
            elif "sigma32" in file_stem:
                desired_value = "sigma32"
            else:
                print("No sigma pattern found in the file name. Defaulting to 'dense array'.")
                desired_value = "dense array"
        else:
            desired_value = "dense array"
    elif parent_dir in meta_mapping:
        desired_value = meta_mapping[parent_dir]
    else:
        print(f"No meta_part_type mapping defined for directory '{parent_dir}'. Skipping update for file {pt_path}.")
        return

    # Check if the first entry already has a meta_part_type key.
    if "meta_part_type" in checkpoint[0]:
        current_value = checkpoint[0]["meta_part_type"]
        if current_value == desired_value:
            print(f"File '{pt_path}' already has 'meta_part_type' set to '{current_value}'. Nothing to update.")
            return
        user_choice = input(
            f"File '{pt_path}' already has 'meta_part_type' in its first entry: currently '{current_value}', "
            f"but desired value is '{desired_value}'. Would you like to [s]kip or [o]verwrite it for all entries? "
        ).strip().lower()
        if user_choice.startswith("s"):
            print(f"Skipping update for this file. Current value '{current_value}' will be retained.")
            return
        elif user_choice.startswith("o"):
            # Overwrite meta_part_type for all entries.
            for entry in checkpoint:
                entry["meta_part_type"] = desired_value
        else:
            print("Invalid choice. Skipping update for this file.")
            return
    else:
        # Add meta_part_type to any entry missing it.
        for entry in checkpoint:
            if "meta_part_type" not in entry:
                entry["meta_part_type"] = desired_value

    # Save the file in place.
    torch.save(checkpoint, pt_path)
    print(f"Updated file '{pt_path}' with meta_part_type='{desired_value}'.")

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
    parser.add_argument(
        "--update-meta", action="store_true",
        help="Update meta_part_type key in .pt file entries based on directory mapping."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    target_path = Path(args.path)

    # If --update-meta flag is provided, update meta_part_type in file(s).
    if args.update_meta:
        if target_path.is_file():
            try:
                validate_pt_file(str(target_path))
            except AssertionError as e:
                print(f"Validation failed for '{target_path}': {e}")
                return
            update_meta_part_type(str(target_path))
        elif target_path.is_dir():
            pt_files = list_all_pt_files(str(target_path))
            if not pt_files:
                print(f"No .pt files found in directory '{target_path}'.")
                return
            for pt_file in pt_files:
                try:
                    validate_pt_file(str(pt_file))
                    update_meta_part_type(str(pt_file))
                except AssertionError as e:
                    print(f"Skipping file '{pt_file}' due to validation error: {e}")
        else:
            print(f"Path '{target_path}' is neither a file nor a directory.")
        return

    # Existing functionality: inspection mode (if a file) or validation mode (if a directory)
    if target_path.is_file():
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
                for k, v in entry.items():
                    print(f"  {k}: {v}")
        except Exception as e:
            print(f"Error inspecting entry: {e}")
    else:
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
