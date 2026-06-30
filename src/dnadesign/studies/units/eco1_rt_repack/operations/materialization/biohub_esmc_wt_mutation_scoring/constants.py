"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_wt_mutation_scoring/constants.py

Constants for Eco1 WT-only ESMC masked-marginal mutation scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT
from dnadesign.thread.adapters.biohub_esmc.client import DEFAULT_BASE_URL

DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
DEFAULT_BIOHUB_API_BASE_URL = DEFAULT_BASE_URL
DEFAULT_BIOHUB_API_VERSION = "v1"
DEFAULT_MODEL = "esmc-300m-2024-12"
DEFAULT_KEY_FILE = Path("../key.md")
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_POSITIONS = "all"
WT_SEQUENCE_ID = "wild_type"

SCORING_RELATIVE_ROOT = Path("biohub_esmc") / "mutation_scoring"
POSITION_ENTROPY_FILE_NAME = "wt_position_entropy.parquet"
SUBSTITUTION_LLR_FILE_NAME = "wt_substitution_llr.parquet"
MASK_JOIN_FILE_NAME = "wt_mutation_scoring_mask_join.parquet"
REQUEST_MANIFEST_FILE_NAME = "wt_mutation_scoring_manifest.yaml"
PLOTS_DIR_NAME = "plots"

REQUEST_MANIFEST_RELATIVE_PATH = "foldcheck_request/foldcheck_request_manifest.yaml"
FOLDCHECK_REPORT_FILE_NAME = "foldcheck_report.parquet"
MASK_SET_FILE_NAME = "mask_set.yaml"
