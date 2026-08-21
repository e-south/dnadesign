"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/tests/test_assessment_publication.py

Atomic, isolated publication of digest-addressed structure assessments.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from dnadesign.contracts.folding import AssessmentTargetV1, StructureAssessmentPolicyV1, StructureAssessmentRequestV1
from dnadesign.contracts.folding.secondary_structure_prediction_v1 import (
    SecondaryStructurePredictionRequestBackendV1,
    SecondaryStructurePredictionRequestDnaPolicyV1,
)
from dnadesign.folding import (
    FoldingConfigError,
    FoldingExecutionError,
    load_published_assessment,
    publish_structure_assessment,
)


def _target() -> AssessmentTargetV1:
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


def _request(*, timeout_seconds: float = 5.0) -> StructureAssessmentRequestV1:
    return StructureAssessmentRequestV1(
        assessment_id="assessment-hop-example",
        target=_target(),
        backend=SecondaryStructurePredictionRequestBackendV1(
            name="ViennaRNA",
            interface="python_api",
            python_module="RNA",
            dna_policy=SecondaryStructurePredictionRequestDnaPolicyV1(mode="convert_t_to_u_for_rna_backend"),
        ),
        policy=StructureAssessmentPolicyV1(required=True, timeout_seconds=timeout_seconds),
    )


def _cli_request(executable: Path, *, timeout_seconds: float) -> StructureAssessmentRequestV1:
    return StructureAssessmentRequestV1(
        assessment_id="assessment-hop-example",
        target=_target(),
        backend=SecondaryStructurePredictionRequestBackendV1(
            name="ViennaRNA",
            interface="cli",
            executable=executable.as_posix(),
            dna_policy=SecondaryStructurePredictionRequestDnaPolicyV1(mode="convert_t_to_u_for_rna_backend"),
        ),
        policy=StructureAssessmentPolicyV1(required=True, timeout_seconds=timeout_seconds),
    )


def _install_fake_rna_module(
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


def test_structure_assessment_publication_round_trips_exact_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"

    published = publish_structure_assessment(_request(), output_dir=output)
    replayed = load_published_assessment(output)

    assert published == replayed
    assert published.record.authority == "advisory"
    assert published.record.target.state_type == "hairpin_encoding_insert"
    assert published.record.prediction.status == "ok"
    assert published.record.prediction.backend is not None
    assert published.record.prediction.backend.version == "test-1.0"


def test_structure_assessment_timeout_leaves_no_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch, delay_seconds=2.0)
    output = tmp_path / "timed-out-assessment"

    with pytest.raises(FoldingExecutionError, match="timed out"):
        publish_structure_assessment(_request(timeout_seconds=0.1), output_dir=output)

    assert not output.exists()


@pytest.mark.skipif(os.name != "posix", reason="process-group timeout contract is POSIX-specific")
def test_structure_assessment_timeout_terminates_cli_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "fake-rnafold"
    descendant_marker = tmp_path / "descendant-survived"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "if '--version' in sys.argv:\n"
        "    print('RNAfold 2.7.2')\n"
        "    raise SystemExit(0)\n"
        "subprocess.Popen([\n"
        "    sys.executable,\n"
        "    '-c',\n"
        "    'import os,pathlib,time; time.sleep(0.5); '"
        '    \'pathlib.Path(os.environ["ASSESSMENT_DESCENDANT_MARKER"]).write_text("alive")\',\n'
        "])\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("ASSESSMENT_DESCENDANT_MARKER", descendant_marker.as_posix())
    output = tmp_path / "timed-out-cli-assessment"

    with pytest.raises(FoldingExecutionError, match="timed out"):
        publish_structure_assessment(
            _cli_request(executable, timeout_seconds=0.1),
            output_dir=output,
        )

    time.sleep(0.7)
    assert not descendant_marker.exists()
    assert not output.exists()


def test_structure_assessment_publication_is_create_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)

    with pytest.raises(FoldingConfigError, match="already exists|create-only"):
        publish_structure_assessment(_request(), output_dir=output)


def test_structure_assessment_loader_rejects_prediction_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    prediction = output / "prediction/secondary_structure_prediction_v1.json"
    prediction.write_bytes(prediction.read_bytes() + b" ")

    with pytest.raises(ValueError, match="prediction digest"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_target_artifact_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    target = output / "assessment-target-sequence.json"
    target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(ValueError, match="target-sequence artifact digest"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_unlisted_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    (output / "unlisted.txt").write_text("not declared\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact inventory"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_symlinked_nested_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    prediction = output / "prediction"
    relocated = output / "prediction-data"
    prediction.rename(relocated)
    prediction.symlink_to(relocated.name, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link"):
        load_published_assessment(output)


def test_structure_assessment_loader_rejects_record_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_rna_module(tmp_path, monkeypatch)
    output = tmp_path / "assessment"
    publish_structure_assessment(_request(), output_dir=output)
    record_path = output / "assessment-record.json"
    manifest_path = output / "manifest.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["assessment_id"] = "different-assessment"
    record_content = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    record_path.write_bytes(record_content)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_digest = f"sha256:{hashlib.sha256(record_content).hexdigest()}"
    manifest["record_digest"] = record_digest
    manifest["artifact_digests"]["assessment-record.json"] = record_digest
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="assessment_id must match"):
        load_published_assessment(output)
