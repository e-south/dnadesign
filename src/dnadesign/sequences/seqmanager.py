"""
--------------------------------------------------------------------------------
<dnadesign project>

You want to ensure each .pt file is valid according to a “contract” (i.e., you define the properties of a valid model state/checkpoint file).
How to do it

Preconditions: For each .pt file, document the shape or keys you expect (e.g., 'model_state_dict', 'optimizer_state_dict', 'epoch', etc.).
Validation: In seqstager.py, have a function like validate_pt_file(file_path) that checks each of those keys and their datatypes.
Exceptions or error handling: If a file doesn’t meet the contract, fail early (e.g., raise an exception or log a message and skip it).
Postconditions: If the file is valid, return a validated data structure.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""


def validate_pt_file(file_path: str):
    """
    Loads a .pt file, checks minimal keys (like `model_state_dict`), 
    and returns True if all checks pass, False (or raises an exception) otherwise.
    """
    checkpoint = torch.load(file_path)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Missing 'model_state_dict' in {file_path}.")
    return True


def check_required_keys(checkpoint: dict, required_keys: list[str]) -> bool:
    for key in required_keys:
        if key not in checkpoint:
            return False
    return True

def check_shapes(checkpoint: dict, expected_shapes: dict):
    # Implementation that compares shapes
    ...
