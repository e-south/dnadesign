"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/study/schema.py

Schema registry and validation entrypoint for Study specs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TypeAlias

from dnadesign.cruncher.study.schema_models import StudyRoot, StudySpec

StudySpecModel: TypeAlias = StudySpec
StudyRootModel: TypeAlias = StudyRoot

STUDY_SCHEMA_ROOT_BY_VERSION: dict[int, type[StudyRootModel]] = {
    3: StudyRoot,
}


def parse_study_root(payload: dict[str, object]) -> StudyRootModel:
    study_payload = payload.get("study")
    if not isinstance(study_payload, dict):
        raise ValueError("Study schema required (study must be a mapping)")
    version = study_payload.get("schema_version")
    root_model = STUDY_SCHEMA_ROOT_BY_VERSION.get(version) if isinstance(version, int) else None
    if root_model is None:
        root_model = StudyRoot
    return root_model.model_validate(payload)
