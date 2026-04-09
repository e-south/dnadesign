"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/payload_resolution.py

Resolve YIU payload sequences from direct input or Sample hit artifacts.
This module is the public orchestration seam; sample-hit IO details live in
`sample_hit_sources.py`.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.yiu.input_payload_models import ResolvedInputPayload
from dnadesign.cruncher.yiu.sample_hit_sources import metadata_text, resolve_sample_hit_payload
from dnadesign.cruncher.yiu.spec_input_models import InputSpec


def resolve_input_payload(input_spec: InputSpec, *, workspace_root: Path, spec_name: str) -> ResolvedInputPayload:
    if input_spec.kind == "user_sequence":
        assert input_spec.user_sequence is not None
        return ResolvedInputPayload(
            input_kind="user_sequence",
            payload_sequence=input_spec.user_sequence.sequence,
            payload_label=None,
            site_label=None,
            provenance={
                "spec_name": spec_name,
                "sample_name": None,
                "hit_id": None,
                "source_artifact_path": None,
            },
            hit_row=None,
            source_artifact_path=None,
            sample_workspace_root=None,
        )

    assert input_spec.sample_hit is not None
    payload_sequence, hit_row, artifact_path, sample_workspace_root = resolve_sample_hit_payload(
        input_spec.sample_hit,
        workspace_root=workspace_root,
    )
    payload_label = (
        metadata_text(input_spec.sample_hit, "payload_label")
        or metadata_text(input_spec.sample_hit, "tf_name")
        or metadata_text(input_spec.sample_hit, "motif_name")
    )
    site_label = metadata_text(input_spec.sample_hit, "site_label")
    provenance = {
        "spec_name": spec_name,
        "sample_name": input_spec.sample_hit.sample_name,
        "hit_id": input_spec.sample_hit.hit_id,
        "source_artifact_path": None if artifact_path is None else str(artifact_path.resolve()),
        "source_workspace": None if sample_workspace_root is None else str(sample_workspace_root.resolve()),
        "metadata": dict(input_spec.sample_hit.metadata),
    }
    return ResolvedInputPayload(
        input_kind="sample_hit",
        payload_sequence=payload_sequence,
        payload_label=payload_label,
        site_label=site_label,
        provenance=provenance,
        hit_row=hit_row,
        source_artifact_path=artifact_path,
        sample_workspace_root=sample_workspace_root,
    )


__all__ = [
    "ResolvedInputPayload",
    "resolve_input_payload",
]
