"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sequence_pseudolikelihood/constants.py

Constants for Eco1 Biohub ESMC sequence pseudo-likelihood scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import re
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT
from dnadesign.thread.adapters.biohub_esmc.client import DEFAULT_BASE_URL, DEFAULT_ESMC_MODEL

DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
DEFAULT_BIOHUB_API_BASE_URL = DEFAULT_BASE_URL
DEFAULT_BIOHUB_API_VERSION = "v1"
DEFAULT_MODEL = DEFAULT_ESMC_MODEL
DEFAULT_KEY_FILE = Path("../key.md")
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_SEQUENCE_LIMIT = "all"
DEFAULT_POSITIONS = "all"
WT_SEQUENCE_ID = "wild_type"

SCORING_RELATIVE_ROOT = Path("biohub_esmc") / "sequence_pseudolikelihood"
POSITION_PLL_FILE_NAME = "position_pll.parquet"
SEQUENCE_PLL_FILE_NAME = "sequence_pll.parquet"
REQUEST_MANIFEST_FILE_NAME = "sequence_pseudolikelihood_manifest.yaml"


def scoring_relative_root_for_model(model: str) -> Path:
    """Return the model-scoped pseudo-likelihood artifact root."""

    model_id = model.strip()
    if not model_id:
        raise ValueError("Biohub ESMC model id must be non-empty")
    if "/" in model_id or "\\" in model_id or ".." in model_id:
        raise ValueError(f"Biohub ESMC model id is not path-safe: {model!r}")
    component = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_").lower()
    if not component:
        raise ValueError(f"Biohub ESMC model id is not path-safe: {model!r}")
    return SCORING_RELATIVE_ROOT / component
