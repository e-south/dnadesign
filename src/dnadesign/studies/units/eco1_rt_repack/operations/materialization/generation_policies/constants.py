"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/generation_policies/constants.py

Constants for Eco1 RT generation-policy materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT

GENERATION_POLICY_VERSION = 3

DISTAL_SCAFFOLD_POLICY_ID = "distal_scaffold_repack_v1"
NEAR_DNA_RNA_ACID_FREE_POLICY_ID = "near_dna_rna_acid_free_v1"
COMBINED_NEAR_PLUS_DISTAL_POLICY_ID = "combined_near_acid_free_plus_distal_v1"

PRIMARY_POLICY_IDS: tuple[str, ...] = (
    DISTAL_SCAFFOLD_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
)

DEFAULT_REQUESTED_VARIANTS_PER_POLICY = 336
DEFAULT_GENERATION_TOTAL_TARGET_RAW = DEFAULT_REQUESTED_VARIANTS_PER_POLICY * len(PRIMARY_POLICY_IDS)

DEFAULT_SOURCE_OUTPUT_ROOT = DEFAULT_THREAD_OUTPUT_ROOT
DEFAULT_GENERATION_POLICIES_ROOT: Path = DEFAULT_THREAD_OUTPUT_ROOT / "generation_policies_v3"
REQUEST_DIR_NAME = "proteinmpnn_request"
POLICY_INPUT_DIR_NAME = "policy_inputs"

CREATED_BY = "dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies"
REQUEST_CREATED_BY = (
    "dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.request_materialization"
)
DEFAULT_CREATED_AT = "2026-07-09T00:00:00Z"

CONSERVATION_PROFILE_ID = "ec86_clade9_conservation_v1"
DIRECT_CONTACT_DISTANCE_ANGSTROM = 5.0
NEAR_REGION_MIN_EXCLUSIVE_ANGSTROM = 5.0
NEAR_REGION_MAX_INCLUSIVE_ANGSTROM = 10.0

MOTIF_CONTEXTS: dict[str, tuple[int, int]] = {
    "naxxh": (99, 115),
    "yadd": (189, 204),
    "vtg": (237, 251),
}
WANG_THUMB_TRACK_POSITIONS: frozenset[int] = frozenset({238, 239, 240, 249, 257, 261, 264, 298})
C_TERMINAL_THUMB_CONTEXT: tuple[int, int] = (255, 311)

STANDARD_AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AMINO_ACIDS_NO_CYS: tuple[str, ...] = tuple(aa for aa in STANDARD_AMINO_ACIDS if aa != "C")
PROTEINMPNN_ALPHABET: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWYX")
ACIDIC_AMINO_ACIDS: frozenset[str] = frozenset({"D", "E"})
PROLINE_GLYCINE_AMINO_ACIDS: frozenset[str] = frozenset({"P", "G"})
TARGET_ALIGNMENT_ROW_ID = "eco1_rt_ec86kit_reference"

PROTEINMPNN_NAME = "chain_a_backbone"
PROTEINMPNN_CHAIN_ID = "A"
PROTEINMPNN_SEED_SET: tuple[int, ...] = (101, 202, 303)
PROTEINMPNN_TEMPERATURES: tuple[float, ...] = (0.1, 0.3)
PROTEINMPNN_BATCH_SIZE = 1

DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
STRUCTURE_SOURCES = DOCS_ROOT / "workbench/provenance/structure-sources.yaml"
