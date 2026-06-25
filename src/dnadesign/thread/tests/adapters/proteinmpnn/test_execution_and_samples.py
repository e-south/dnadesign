"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/tests/adapters/proteinmpnn/test_execution_and_samples.py

ProteinMPNN execution preflight and sample parsing tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.thread.adapters.proteinmpnn.execution import ProteinMpnnExecutionConfig
from dnadesign.thread.adapters.proteinmpnn.execution_preflight import validate_proteinmpnn_root
from dnadesign.thread.adapters.proteinmpnn.samples import parse_proteinmpnn_fasta_samples


def test_proteinmpnn_preflight_rejects_missing_official_scripts(tmp_path: Path) -> None:
    issues = validate_proteinmpnn_root(tmp_path)

    check_ids = {issue.check_id for issue in issues}
    assert "thread.proteinmpnn.tool_missing_script" in check_ids
    assert "thread.proteinmpnn.tool_missing_weights" in check_ids


def test_proteinmpnn_preflight_accepts_required_scripts_and_weights(tmp_path: Path) -> None:
    for rel_path in (
        "protein_mpnn_run.py",
        "helper_scripts/parse_multiple_chains.py",
        "helper_scripts/assign_fixed_chains.py",
        "helper_scripts/make_fixed_positions_dict.py",
        "vanilla_model_weights/v_48_020.pt",
    ):
        path = tmp_path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")

    assert validate_proteinmpnn_root(tmp_path) == []


def test_parse_proteinmpnn_fasta_samples_extracts_seed_temperature_and_scores(tmp_path: Path) -> None:
    fasta_path = tmp_path / "seqs" / "chain_a_backbone.fa"
    fasta_path.parent.mkdir()
    fasta_path.write_text(
        "\n".join(
            [
                (
                    ">chain_a_backbone, score=1.0000, global_score=2.0000, fixed_chains=[], "
                    "designed_chains=['A'], model_name=v_48_020, git_hash=abc123, seed=101"
                ),
                "AAA",
                ">T=0.1, sample=1, score=0.5000, global_score=1.5000, seq_recovery=0.9000",
                "AAC",
                ">T=0.3, sample=1, score=0.7000, global_score=1.7000, seq_recovery=0.8000",
                "AAD",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = parse_proteinmpnn_fasta_samples(
        fasta_path,
        backend_run_id="run-1",
        request_hash="sha256:request",
        seed=101,
        sequence_length=3,
    )

    assert [row["temperature"] for row in rows] == [0.1, 0.3]
    assert [row["sequence"] for row in rows] == ["AAC", "AAD"]
    assert rows[0]["sample_id"] == "run-1__seed101__temp0.1__sample1"
    assert rows[0]["score"] == 0.5
    assert rows[0]["global_score"] == 1.5
    assert rows[0]["status"] == "accepted"


def test_proteinmpnn_execution_config_derives_expected_samples() -> None:
    config = ProteinMpnnExecutionConfig(batch_id="eco1_rt_n96", num_seq_per_target=16, batch_size=4)

    assert config.expected_sample_count(seed_count=3, temperature_count=2) == 96
    assert config.run_dir_name == "eco1_rt_n96"


def test_proteinmpnn_execution_config_rejects_nondivisible_batch_size() -> None:
    with pytest.raises(ValueError, match="num_seq_per_target must be divisible"):
        ProteinMpnnExecutionConfig(batch_id="bad", num_seq_per_target=10, batch_size=3)
