"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/snapback_released_solve_snapshot.py

Request snapshot helpers for released-product Snapback solve runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import yaml

from dnadesign.cruncher.snapback.released_models import (
    ReleasedSolveOutputConfig,
    SingleNickReleasedTargetSearchRequest,
)


def build_released_solve_request_snapshot_payload(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    output: ReleasedSolveOutputConfig,
) -> dict[str, object]:
    return {
        "released_solve": {
            "schema_version": 1,
            "kind": "single_nick_released_solve_v1",
        },
        "target": request.target.model_dump(mode="json"),
        "nick_sources": {
            "preset": request.nick_sources.preset,
            "additional_presets": list(request.nick_sources.additional_presets),
            "additional_paths": [str(path) for path in request.nick_sources.additional_paths],
        },
        "release_sources": {
            "preset": request.release_sources.preset,
            "additional_presets": list(request.release_sources.additional_presets),
            "additional_paths": [str(path) for path in request.release_sources.additional_paths],
        },
        "search": request.search.model_dump(mode="json"),
        "output": output.model_dump(mode="json"),
    }


def dump_released_solve_request_snapshot_yaml(
    *,
    request: SingleNickReleasedTargetSearchRequest,
    output: ReleasedSolveOutputConfig,
) -> str:
    return yaml.safe_dump(
        build_released_solve_request_snapshot_payload(request=request, output=output),
        sort_keys=False,
    )


__all__ = [
    "build_released_solve_request_snapshot_payload",
    "dump_released_solve_request_snapshot_yaml",
]
