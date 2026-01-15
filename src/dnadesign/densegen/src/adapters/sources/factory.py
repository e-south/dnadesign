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
    PWMJasparInput,
    PWMMatrixCSVInput,
    PWMMemeInput,
    SequenceLibraryInput,
    USRSequencesInput,
)
from .base import BaseDataSource
from .binding_sites import BindingSitesDataSource
from .pwm_jaspar import PWMJasparDataSource
from .pwm_matrix_csv import PWMMatrixCSVDataSource
from .pwm_meme import PWMMemeDataSource
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
        )
    if isinstance(cfg, PWMJasparInput):
        return PWMJasparDataSource(
            path=cfg.path,
            cfg_path=cfg_path,
            motif_ids=cfg.motif_ids,
            sampling=cfg.sampling.model_dump(),
        )
    if isinstance(cfg, PWMMatrixCSVInput):
        return PWMMatrixCSVDataSource(
            path=cfg.path,
            cfg_path=cfg_path,
            motif_id=cfg.motif_id,
            columns=cfg.columns.model_dump(),
            sampling=cfg.sampling.model_dump(),
        )
    if isinstance(cfg, USRSequencesInput):
        return USRSequencesDataSource(dataset=cfg.dataset, cfg_path=cfg_path, root=cfg.root, limit=cfg.limit)
    raise ValueError(f"Unsupported source config type: {type(cfg)}")
