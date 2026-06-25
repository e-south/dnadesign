"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/sampling/test_sample_table.py

Sample-table contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling import validate_sample_table_content
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.mask_set import materialize_mask_set
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_request import (
    materialize_proteinmpnn_request,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_sample_ingest import (
    materialize_proteinmpnn_samples,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.thread_plan import materialize_thread_plan
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.mask_set._fixtures import (
    materialize_upstream_artifacts,
)
from dnadesign.thread.adapters.proteinmpnn.samples import write_backend_run_manifest

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(),
    reason="requires sibling ec86kit structure-authority artifacts",
)


def test_sample_table_contract_rejects_request_hash_drift(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    materialize_thread_plan(repo_root=repo_root(), output_root=tmp_path)
    materialize_proteinmpnn_request(repo_root=repo_root(), output_root=tmp_path)
    result = materialize_proteinmpnn_samples(
        repo_root=repo_root(),
        output_root=tmp_path,
        proteinmpnn_root=tmp_path / "fake_proteinmpnn",
        runner=_fake_runner,
    )

    table = pq.read_table(result.sample_table_path)
    rows = table.to_pylist()
    rows[0]["request_hash"] = "sha256:stale-request"
    pq.write_table(pa.Table.from_pylist(rows).replace_schema_metadata(table.schema.metadata), result.sample_table_path)

    issues = validate_sample_table_content(result.sample_table_path, output_root=tmp_path)

    assert {issue.check_id for issue in issues} == {"eco1_rt.sampling.sample_table_request_hash_mismatch"}


def _fake_runner(
    *,
    request_manifest_path: Path,
    proteinmpnn_root: Path,
    output_dir: Path,
    execution_config: Any,
) -> dict[str, Any]:
    del proteinmpnn_root
    manifest = yaml.safe_load(request_manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    parsed_path = Path(str(manifest["sidecar_paths"]["parsed_pdbs_jsonl"]))
    wt_record = json.loads(parsed_path.read_text(encoding="utf-8").splitlines()[0])
    wt_sequence = str(wt_record["seq_chain_A"])
    mutable_positions = [int(position) for position in manifest["mutable_positions_by_chain"]["A"]]
    rows: list[dict[str, Any]] = []
    batch_dir = output_dir / "batches" / execution_config.batch_id
    run_manifest_path = batch_dir / "backend_run_manifest.yaml"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for seed in manifest["seed_set"]:
        seed_dir = batch_dir / f"seed_{seed}"
        fasta = seed_dir / "seqs" / "chain_a_backbone.fa"
        fasta.parent.mkdir(parents=True, exist_ok=True)
        records = [
            (
                f">chain_a_backbone, score=1.0000, global_score=2.0000, fixed_chains=[], "
                f"designed_chains=['A'], model_name=v_48_020, git_hash=fake, seed={seed}"
            ),
            wt_sequence,
        ]
        for temperature in manifest["temperature_schedule"]:
            for sample_index in range(1, int(manifest["num_seq_per_target"]) + 1):
                sequence = list(wt_sequence)
                position = mutable_positions[(sample_index - 1) % len(mutable_positions)] - 1
                sequence[position] = "K" if sequence[position] != "K" else "A"
                records.extend(
                    [
                        (
                            f">T={temperature:g}, sample={sample_index}, score=0.5000, "
                            "global_score=1.5000, seq_recovery=0.9000"
                        ),
                        "".join(sequence),
                    ]
                )
        fasta.write_text("\n".join(records) + "\n", encoding="utf-8")
        rows.append(
            {
                "seed": seed,
                "output_dir": str(seed_dir),
                "returncode": 0,
                "stdout": "",
                "stderr": "",
            }
        )
    write_backend_run_manifest(
        run_manifest_path,
        request_manifest_path=request_manifest_path,
        request_hash=str(manifest["request_hash"]),
        proteinmpnn_root=output_dir,
        proteinmpnn_git_commit="fake",
        runs=rows,
        batch_id=execution_config.batch_id,
        num_seq_per_target=execution_config.num_seq_per_target,
        batch_size=execution_config.batch_size,
        expected_sample_count=int(manifest["expected_sample_count"]),
    )
    return {
        "backend_run_manifest_path": run_manifest_path,
        "backend_run_id": "proteinmpnn_fake",
        "request_hash": manifest["request_hash"],
        "run_outputs": rows,
    }
