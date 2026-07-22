"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_report/constants.py

Constants for Eco1 fold-check report materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

DEFAULT_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
REQUEST_MANIFEST_RELATIVE_PATH = "foldcheck_request/foldcheck_request_manifest.yaml"
REFERENCE_BACKBONE_RELATIVE_PATH = "proteinmpnn_request/chain_a_backbone.pdb"
RESIDUE_MAP_FILE_NAME = "residue_map.parquet"
REPORT_FILE_NAME = "foldcheck_report.parquet"
DEFAULT_RUNTIME_PARAMETERS = {
    "command": "colabfold_batch",
    "execution_locus": "external_colabfold_cli",
}
