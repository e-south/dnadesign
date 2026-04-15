"""
Sample services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..io.hashing import sha256_file
from ..io.manifest_io import write_manifest
from ..runs.recorder import record_audit
from ..samples.build import build_sample_artifact
from ..version import __version__
from ..workspaces.loader import load_workspace_config


def build_sample(
    workspace: str | Path,
    sample_id: str,
    *,
    view_id: str | None,
    strategy: str,
    n: int | None,
    group_column: str | None,
    seed: int,
    reference_set_id: str | None = None,
    explicit_ids: list[str] | None = None,
    input_sample_ids: list[str] | None = None,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    sample_dir = context.output_root / "samples" / sample_id
    if sample_dir.exists() and not force:
        raise ArtifactConflictError(f"sample artifact already exists: {sample_dir}")
    if force and sample_dir.exists():
        import shutil

        shutil.rmtree(sample_dir)

    artifact_dir, rows = build_sample_artifact(
        context,
        sample_id=sample_id,
        view_id=view_id,
        strategy=strategy,
        n=n,
        group_column=group_column,
        seed=seed,
        reference_set_id=reference_set_id,
        explicit_ids=explicit_ids,
        input_sample_ids=input_sample_ids,
    )
    input_entries: list[ArtifactInput]
    input_digests: dict[str, str]
    if strategy in {"union", "intersection"}:
        input_entries = []
        input_digests = {}
        for input_sample_id in input_sample_ids or []:
            input_path = context.output_root / "samples" / input_sample_id / "rows.parquet"
            input_digest = sha256_file(input_path)
            input_entries.append(
                ArtifactInput(
                    kind="sample_set",
                    id=input_sample_id,
                    digest=input_digest,
                    path=input_path.as_posix(),
                )
            )
            input_digests[f"sample_set:{input_sample_id}"] = input_digest
    else:
        assert view_id is not None
        input_path = context.output_root / "views" / view_id / "rows.parquet"
        input_digest = sha256_file(input_path)
        input_entries = [ArtifactInput(kind="view_rows", id=view_id, digest=input_digest, path=input_path.as_posix())]
        input_digests = {"view_rows": input_digest}
    manifest = ArtifactManifest(
        artifact_kind="sample_set",
        artifact_id=sample_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="sample build",
        inputs=input_entries,
        input_digests=input_digests,
        freshness_basis={"kind": "artifact_inputs", "known": True},
        params={
            "strategy": strategy,
            "n": n,
            "group_column": group_column,
            "seed": seed,
            "reference_set": reference_set_id,
            "explicit_ids": explicit_ids or [],
            "input_sample_ids": input_sample_ids or [],
        },
        outputs=[ArtifactOutput(path="rows.parquet", media_type="application/x-parquet")],
        stats={"rows": rows},
    )
    write_manifest(artifact_dir / "manifest.json", manifest.model_dump(mode="json"))
    result = CommandResult(
        command="sample build",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="sample_set",
        artifact_id=sample_id,
        outputs=[artifact_dir.as_posix()],
        inputs={
            "view": view_id,
            "reference_set": reference_set_id,
            "explicit_ids": explicit_ids or [],
            "input_samples": input_sample_ids or [],
        },
        input_digests=input_digests,
        metrics={"rows": rows},
        freshness_known=True,
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="sample_build",
        artifact_id=sample_id,
    )
    return result
