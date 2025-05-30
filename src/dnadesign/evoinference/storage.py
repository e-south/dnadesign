"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/evoinference/storage.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os

import torch
import yaml
from logger import get_logger

logger = get_logger(__name__)


def write_results(filepath: str, data: list, overwrite: bool):
    """
    Write the updated list of dictionaries back to the .pt file.
    It is assumed that the data entries have been updated in memory.
    """
    try:
        torch.save(data, filepath)
        logger.info(f"Results written to {filepath}")
    except Exception as e:
        logger.error(f"Error writing results to {filepath}: {str(e)}")
        raise e


def update_progress(progress_filepath: str, progress_data: dict):
    """
    Write progress summary data to a YAML file.
    """
    try:
        with open(progress_filepath, "w") as f:
            yaml.safe_dump(progress_data, f)
        logger.info(f"Progress updated in {progress_filepath}")
    except Exception as e:
        logger.error(f"Error writing progress file {progress_filepath}: {str(e)}")
        raise e


def load_progress(progress_filepath: str):
    """
    Load progress data from a YAML file, if it exists; otherwise, return an empty dict.
    """
    if os.path.exists(progress_filepath):
        try:
            with open(progress_filepath, "r") as f:
                data = yaml.safe_load(f)
            return data if data is not None else {}
        except Exception as e:
            logger.error(f"Error loading progress file {progress_filepath}: {str(e)}")
            return {}
    else:
        return {}
