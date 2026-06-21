from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.aligner.msa import MsaBackendSpec, MsaRequest, load_fasta_records, run_msa
from dnadesign.aligner.msa.backends.mafft import MafftUnavailableError, preflight_mafft


def test_mafft_preflight_fails_without_executable() -> None:
    spec = MsaBackendSpec(backend_id="mafft", executable="definitely_missing_mafft_binary")

    with pytest.raises(MafftUnavailableError, match="MAFFT executable not found"):
        preflight_mafft(spec)


def test_run_msa_writes_aligned_fasta_and_manifest(tmp_path: Path) -> None:
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    manifest_path = tmp_path / "aligned.manifest.yaml"
    input_fasta.write_text(">target\nACDE\n>other\nACDF\n", encoding="utf-8")

    request = MsaRequest(
        input_fasta=input_fasta,
        output_fasta=output_fasta,
        manifest_path=manifest_path,
        target_row_id="target",
        backend=MsaBackendSpec(backend_id="mafft"),
        command_args=("--auto",),
    )

    try:
        result = run_msa(request)
    except MafftUnavailableError:
        pytest.skip("MAFFT is not installed in the current execution environment")

    assert result.aligned_fasta == output_fasta
    assert output_fasta.exists()
    assert manifest_path.exists()
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "backend_id: mafft" in manifest_text
    assert "target_row_id: target" in manifest_text
    assert "input_fasta_sha256:" in manifest_text
    assert "output_fasta_sha256:" in manifest_text
    assert "elapsed_seconds:" in manifest_text
    assert "return_code: 0" in manifest_text
    assert "stderr_path:" in manifest_text


def test_run_msa_publishes_output_only_after_valid_backend_result(tmp_path: Path) -> None:
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    manifest_path = tmp_path / "aligned.manifest.yaml"
    stderr_path = tmp_path / "aligned.stderr.txt"
    input_fasta.write_text(">target\nACDE\n>other\nACDF\n", encoding="utf-8")
    fake_mafft = _write_fake_mafft(
        tmp_path / "fake-mafft",
        """
        if [ "$1" = "--version" ]; then
          echo "v7.526"
          exit 0
        fi
        cat "$2"
        """,
    )

    result = run_msa(
        MsaRequest(
            input_fasta=input_fasta,
            output_fasta=output_fasta,
            manifest_path=manifest_path,
            target_row_id="target",
            backend=MsaBackendSpec(executable=str(fake_mafft)),
            command_args=("--auto",),
            stderr_path=stderr_path,
            run_label="unit-profile",
        )
    )

    assert result.aligned_fasta == output_fasta
    assert result.elapsed_seconds >= 0
    assert result.return_code == 0
    assert result.stderr_path == stderr_path
    assert load_fasta_records(output_fasta, alphabet="protein", allow_gaps=True)["target"] == "ACDE"
    assert not list(tmp_path.glob(".aligned.fasta.*.tmp"))
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_label"] == "unit-profile"
    assert manifest["elapsed_seconds"] >= 0
    assert manifest["return_code"] == 0
    assert manifest["stderr_path"] == str(stderr_path)
    assert manifest["stderr_sha256"].startswith("sha256:")


def test_run_msa_removes_partial_temp_output_when_backend_fails(tmp_path: Path) -> None:
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    manifest_path = tmp_path / "aligned.manifest.yaml"
    stderr_path = tmp_path / "aligned.stderr.txt"
    input_fasta.write_text(">target\nACDE\n>other\nACDF\n", encoding="utf-8")
    fake_mafft = _write_fake_mafft(
        tmp_path / "fake-mafft",
        """
        if [ "$1" = "--version" ]; then
          echo "v7.526"
          exit 0
        fi
        echo ">target"
        echo "PARTIAL"
        echo "backend failure" >&2
        exit 2
        """,
    )

    with pytest.raises(RuntimeError, match="MAFFT failed with exit code 2"):
        run_msa(
            MsaRequest(
                input_fasta=input_fasta,
                output_fasta=output_fasta,
                manifest_path=manifest_path,
                target_row_id="target",
                backend=MsaBackendSpec(executable=str(fake_mafft)),
                command_args=("--auto",),
                stderr_path=stderr_path,
            )
        )

    assert not output_fasta.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.glob(".aligned.fasta.*.tmp"))
    assert "backend failure" in stderr_path.read_text(encoding="utf-8")


def test_run_msa_removes_partial_temp_output_when_backend_times_out(tmp_path: Path) -> None:
    input_fasta = tmp_path / "input.fasta"
    output_fasta = tmp_path / "aligned.fasta"
    manifest_path = tmp_path / "aligned.manifest.yaml"
    input_fasta.write_text(">target\nACDE\n>other\nACDF\n", encoding="utf-8")
    fake_mafft = _write_fake_mafft(
        tmp_path / "fake-mafft",
        """
        if [ "$1" = "--version" ]; then
          echo "v7.526"
          exit 0
        fi
        echo ">target"
        echo "PARTIAL"
        sleep 2
        """,
    )

    with pytest.raises(TimeoutError, match="MAFFT timed out"):
        run_msa(
            MsaRequest(
                input_fasta=input_fasta,
                output_fasta=output_fasta,
                manifest_path=manifest_path,
                target_row_id="target",
                backend=MsaBackendSpec(executable=str(fake_mafft)),
                command_args=("--auto",),
                timeout_seconds=0.1,
            )
        )

    assert not output_fasta.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.glob(".aligned.fasta.*.tmp"))


def _write_fake_mafft(path: Path, body: str) -> Path:
    script = "#!/bin/sh\nset -eu\n" + "\n".join(line.strip() for line in body.strip().splitlines()) + "\n"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o755)
    return path
