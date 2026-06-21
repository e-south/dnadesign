"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/conservation_alignments/test_command_policy.py

Declared MSA command-policy tests for Eco1 conservation alignments.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.conservation_alignments import (
    parse_declared_alignment_command,
    parse_declared_mafft_args,
)


def test_declared_mafft_args_rejects_silent_command_drift() -> None:
    with pytest.raises(ValueError, match="must use '<input_fasta> > <output_fasta>'"):
        parse_declared_mafft_args("mafft --auto <input_fasta>")


def test_declared_alignment_command_parses_clustalo_placeholders() -> None:
    backend_id, command_args = parse_declared_alignment_command(
        "clustalo --force --outfmt=fasta --threads=1 -i <input_fasta> -o <output_fasta>"
    )

    assert backend_id == "clustalo"
    assert command_args == ("--force", "--outfmt=fasta", "--threads=1")
