"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/biohub_esmc_sae_profile/constants.py

Constants for Eco1 Biohub ESMC SAE-profile materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT
from dnadesign.thread.adapters.biohub_esmc.client import (
    BIOHUB_API_VERSION,
    DEFAULT_BASE_URL,
    DEFAULT_ESMC_MODEL,
    DEFAULT_ESMC_SAE_MODEL,
)

DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
REQUEST_MANIFEST_RELATIVE_PATH = "foldcheck_request/foldcheck_request_manifest.yaml"
FOLDCHECK_REPORT_FILE_NAME = "foldcheck_report.parquet"
DEFAULT_SEQUENCE_LIMIT = "1"
DEFAULT_BIOHUB_API_BASE_URL = DEFAULT_BASE_URL
DEFAULT_BIOHUB_API_VERSION = BIOHUB_API_VERSION
DEFAULT_MODEL = DEFAULT_ESMC_MODEL
DEFAULT_SAE_MODEL = DEFAULT_ESMC_SAE_MODEL
DEFAULT_NORMALIZE_FEATURES = False
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_KEY_FILE = Path("../key.md")
