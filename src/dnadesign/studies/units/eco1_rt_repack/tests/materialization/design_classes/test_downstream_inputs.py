"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/design_classes/test_downstream_inputs.py

Design-class downstream-input staging tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes import (
    materialize_design_class_candidate_pool,
    materialize_design_class_downstream_inputs,
    materialize_design_class_requests,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.cli import main
from dnadesign.thread.candidates import write_candidate_table


def test_downstream_inputs_stage_shared_review_inputs_without_root_mask(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "classes"
    _write_shared_downstream_inputs(source_root)
    materialize_design_class_requests(repo_root=Path.cwd(), output_root=output_root)
    baseline_root = tmp_path / "baseline"
    _write_candidate_table(
        baseline_root / "candidate_table.parquet",
        [_candidate("thread_candidate_a", "sha256:aaa", rank=1)],
    )
    materialize_design_class_candidate_pool(
        repo_root=Path.cwd(),
        output_root=output_root,
        source_output_root=source_root,
        baseline_candidate_table_path=baseline_root / "candidate_table.parquet",
    )

    result = materialize_design_class_downstream_inputs(
        repo_root=Path.cwd(),
        output_root=output_root,
        source_output_root=source_root,
    )
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))

    assert result.candidate_table_path.exists()
    assert pq.read_table(result.candidate_table_path).num_rows == 1
    assert (output_root / "residue_map.parquet").read_text(encoding="utf-8") == "residue-map\n"
    assert (output_root / "proteinmpnn_request/chain_a_backbone.pdb").read_text(encoding="utf-8") == "pdb\n"
    assert (output_root / "biohub_esmc/mutation_scoring/wt_mutation_scoring_manifest.yaml").exists()
    assert not (output_root / "mask_set.yaml").exists()
    assert "No root-level mask_set.yaml" in manifest["mask_policy_note"]


def test_downstream_inputs_cli_reports_paths(tmp_path: Path, capsys) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "classes"
    _write_shared_downstream_inputs(source_root)
    materialize_design_class_requests(repo_root=Path.cwd(), output_root=output_root)
    baseline_root = tmp_path / "baseline"
    _write_candidate_table(
        baseline_root / "candidate_table.parquet",
        [_candidate("thread_candidate_a", "sha256:aaa", rank=1)],
    )
    materialize_design_class_candidate_pool(
        repo_root=Path.cwd(),
        output_root=output_root,
        source_output_root=source_root,
        baseline_candidate_table_path=baseline_root / "candidate_table.parquet",
    )

    exit_code = main(
        [
            "--repo-root",
            str(Path.cwd()),
            "--output-root",
            str(output_root),
            "--source-output-root",
            str(source_root),
            "downstream-inputs",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "downstream_inputs_manifest_path" in captured.out
    assert "candidate_table_path" in captured.out


def _write_candidate_table(path: Path, rows: list[dict[str, object]]) -> None:
    write_candidate_table(path, rows, request_hash="sha256:test")


def _write_shared_downstream_inputs(source_root: Path) -> None:
    files = {
        "residue_map.parquet": "residue-map\n",
        "conservation_profile.parquet": "conservation\n",
        "proteinmpnn_request/chain_a_backbone.pdb": "pdb\n",
        "proteinmpnn_request/request_manifest.yaml": "request: ok\n",
        "conservation_alignments/ec86_clade9_conservation_v1.aligned.fasta": ">a\nAA\n",
        "conservation_sources/ec86_clade9_conservation_v1.source_manifest.yaml": "source: ok\n",
        "biohub_esmc/mutation_scoring/wt_mutation_scoring_manifest.yaml": "model: esmc-300m-2024-12\n",
        "biohub_esmc/mutation_scoring/wt_substitution_llr.parquet": "llr\n",
    }
    for relative_path, content in files.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _candidate(candidate_id: str, sequence_hash: str, *, rank: int) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "source_sample_id": f"{candidate_id}_sample",
        "backend_run_id": "backend",
        "request_hash": "sha256:test",
        "sequence_hash": sequence_hash,
        "sequence": "ACDE",
        "score": float(rank),
        "global_score": float(rank),
        "seq_recovery": 0.5,
        "seed": 101,
        "temperature": 0.1,
        "sample_index": rank,
        "duplicate_sample_count": 1,
        "mutation_count": 1,
        "mutable_mutation_count": 1,
        "protected_mutation_count": 0,
        "outside_mutable_positions": [],
        "canonical_mutations": ["A1C"],
        "status": "accepted",
        "rank": rank,
    }
