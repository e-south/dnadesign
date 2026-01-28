"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/factory.py

Factory for DenseGen input sources.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ...config import (
    BindingSitesInput,
    PWMArtifactInput,
    PWMArtifactSetInput,
    PWMJasparInput,
    PWMMatrixCSVInput,
    PWMMemeInput,
    PWMMemeSetInput,
    SequenceLibraryInput,
    USRSequencesInput,
)
from .base import BaseDataSource
from .binding_sites import BindingSitesDataSource
from .pwm_artifact import PWMArtifactDataSource
from .pwm_artifact_set import PWMArtifactSetDataSource
from .pwm_jaspar import PWMJasparDataSource
from .pwm_matrix_csv import PWMMatrixCSVDataSource
from .pwm_meme import PWMMemeDataSource
from .pwm_meme_set import PWMMemeSetDataSource
from .sequence_library import SequenceLibraryDataSource
from .usr_sequences import USRSequencesDataSource


def data_source_factory(cfg, cfg_path: Path) -> BaseDataSource:
    if isinstance(cfg, BindingSitesInput):
        return BindingSitesDataSource(
            path=cfg.path,
            cfg_path=cfg_path,
            fmt=cfg.format,
            columns=cfg.columns.model_dump(),
        )
    if isinstance(cfg, SequenceLibraryInput):
        return SequenceLibraryDataSource(
            path=cfg.path,
            cfg_path=cfg_path,
            fmt=cfg.format,
            sequence_column=cfg.sequence_column,
        )
    if isinstance(cfg, PWMMemeInput):
        return PWMMemeDataSource(
            path=cfg.path,
            cfg_path=cfg_path,
            motif_ids=cfg.motif_ids,
            sampling=cfg.sampling.model_dump(),
            input_name=cfg.name,
        )
    if isinstance(cfg, PWMMemeSetInput):
        return PWMMemeSetDataSource(
            paths=list(cfg.paths),
            cfg_path=cfg_path,
            motif_ids=cfg.motif_ids,
            sampling=cfg.sampling.model_dump(),
            input_name=cfg.name,
        )
    if isinstance(cfg, PWMJasparInput):
        return PWMJasparDataSource(
            path=cfg.path,
            cfg_path=cfg_path,
            motif_ids=cfg.motif_ids,
            sampling=cfg.sampling.model_dump(),
            input_name=cfg.name,
        )
    if isinstance(cfg, PWMMatrixCSVInput):
        return PWMMatrixCSVDataSource(
            path=cfg.path,
            cfg_path=cfg_path,
            motif_id=cfg.motif_id,
            columns=cfg.columns.model_dump(),
            sampling=cfg.sampling.model_dump(),
            input_name=cfg.name,
        )
    if isinstance(cfg, PWMArtifactInput):
        return PWMArtifactDataSource(
            path=cfg.path,
            cfg_path=cfg_path,
            sampling=cfg.sampling.model_dump(),
            input_name=cfg.name,
        )
    if isinstance(cfg, PWMArtifactSetInput):
        return PWMArtifactSetDataSource(
            paths=list(cfg.paths),
            cfg_path=cfg_path,
            sampling=cfg.sampling.model_dump(),
            overrides_by_motif_id={k: v.model_dump() for k, v in cfg.overrides_by_motif_id.items()},
            input_name=cfg.name,
        )
    if isinstance(cfg, USRSequencesInput):
        return USRSequencesDataSource(dataset=cfg.dataset, cfg_path=cfg_path, root=cfg.root, limit=cfg.limit)
    raise ValueError(f"Unsupported source config type: {type(cfg)}")
