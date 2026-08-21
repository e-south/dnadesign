"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/tests/_assessment_fixtures.py

Shared fixtures for digest-addressed structure assessment tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from dnadesign.contracts.folding import AssessmentTargetV1, StructureAssessmentPolicyV1, StructureAssessmentRequestV1
from dnadesign.contracts.folding.secondary_structure_prediction_v1 import (
    SecondaryStructurePredictionRequestBackendV1,
    SecondaryStructurePredictionRequestDnaPolicyV1,
)


def assessment_target() -> AssessmentTargetV1:
    sequence = "GCATGC"
    return AssessmentTargetV1(
        state_id="hop:encoding/example",
        state_type="hairpin_encoding_insert",
        state_schema="hop.plan/v2",
        state_digest=f"sha256:{'1' * 64}",
        sequence_id="hop:encoding/example",
        sequence_sha256=f"sha256:{hashlib.sha256(sequence.encode()).hexdigest()}",
        sequence=sequence,
        strandedness="not_asserted",
        topology="not_asserted",
    )


def assessment_request(*, timeout_seconds: float = 5.0) -> StructureAssessmentRequestV1:
    return StructureAssessmentRequestV1(
        assessment_id="assessment-hop-example",
        target=assessment_target(),
        backend=SecondaryStructurePredictionRequestBackendV1(
            name="ViennaRNA",
            interface="python_api",
            python_module="RNA",
            dna_policy=SecondaryStructurePredictionRequestDnaPolicyV1(mode="convert_t_to_u_for_rna_backend"),
        ),
        policy=StructureAssessmentPolicyV1(required=True, timeout_seconds=timeout_seconds),
    )


def cli_assessment_request(executable: Path, *, timeout_seconds: float) -> StructureAssessmentRequestV1:
    return StructureAssessmentRequestV1(
        assessment_id="assessment-hop-example",
        target=assessment_target(),
        backend=SecondaryStructurePredictionRequestBackendV1(
            name="ViennaRNA",
            interface="cli",
            executable=executable.as_posix(),
            dna_policy=SecondaryStructurePredictionRequestDnaPolicyV1(mode="convert_t_to_u_for_rna_backend"),
        ),
        policy=StructureAssessmentPolicyV1(required=True, timeout_seconds=timeout_seconds),
    )


def install_fake_rna_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    delay_seconds: float = 0.0,
) -> None:
    module_root = tmp_path / "fake-backend"
    module_root.mkdir()
    (module_root / "RNA.py").write_text(
        "__version__ = 'test-1.0'\n"
        "import time\n"
        "class Compound:\n"
        "    def mfe(self):\n"
        f"        time.sleep({delay_seconds!r})\n"
        "        return '((..))', -1.2\n"
        "def fold_compound(sequence):\n"
        "    return Compound()\n",
        encoding="utf-8",
    )
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(part for part in (module_root.as_posix(), existing) if part))


__all__ = [
    "assessment_request",
    "assessment_target",
    "cli_assessment_request",
    "install_fake_rna_module",
]
