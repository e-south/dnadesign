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
from typing import Callable

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


def _build_binding_sites(cfg: BindingSitesInput, cfg_path: Path) -> BaseDataSource:
    return BindingSitesDataSource(
        path=cfg.path,
        cfg_path=cfg_path,
        fmt=cfg.format,
        columns=cfg.columns.model_dump(),
    )


def _build_sequence_library(cfg: SequenceLibraryInput, cfg_path: Path) -> BaseDataSource:
    return SequenceLibraryDataSource(
        path=cfg.path,
        cfg_path=cfg_path,
        fmt=cfg.format,
        sequence_column=cfg.sequence_column,
    )


def _build_pwm_meme(cfg: PWMMemeInput, cfg_path: Path) -> BaseDataSource:
    return PWMMemeDataSource(
        path=cfg.path,
        cfg_path=cfg_path,
        motif_ids=cfg.motif_ids,
        sampling=cfg.sampling,
        input_name=cfg.name,
    )


def _build_pwm_meme_set(cfg: PWMMemeSetInput, cfg_path: Path) -> BaseDataSource:
    return PWMMemeSetDataSource(
        paths=list(cfg.paths),
        cfg_path=cfg_path,
        motif_ids=cfg.motif_ids,
        sampling=cfg.sampling,
        input_name=cfg.name,
    )


def _build_pwm_jaspar(cfg: PWMJasparInput, cfg_path: Path) -> BaseDataSource:
    return PWMJasparDataSource(
        path=cfg.path,
        cfg_path=cfg_path,
        motif_ids=cfg.motif_ids,
        sampling=cfg.sampling,
        input_name=cfg.name,
    )


def _build_pwm_matrix_csv(cfg: PWMMatrixCSVInput, cfg_path: Path) -> BaseDataSource:
    return PWMMatrixCSVDataSource(
        path=cfg.path,
        cfg_path=cfg_path,
        motif_id=cfg.motif_id,
        columns=cfg.columns.model_dump(),
        sampling=cfg.sampling,
        input_name=cfg.name,
    )


def _build_pwm_artifact(cfg: PWMArtifactInput, cfg_path: Path) -> BaseDataSource:
    return PWMArtifactDataSource(
        path=cfg.path,
        cfg_path=cfg_path,
        sampling=cfg.sampling,
        input_name=cfg.name,
    )


def _build_pwm_artifact_set(cfg: PWMArtifactSetInput, cfg_path: Path) -> BaseDataSource:
    return PWMArtifactSetDataSource(
        paths=list(cfg.paths),
        cfg_path=cfg_path,
        sampling=cfg.sampling,
        overrides_by_motif_id=dict(cfg.overrides_by_motif_id),
        input_name=cfg.name,
    )


def _build_usr_sequences(cfg: USRSequencesInput, cfg_path: Path) -> BaseDataSource:
    return USRSequencesDataSource(dataset=cfg.dataset, cfg_path=cfg_path, root=cfg.root, limit=cfg.limit)


_SOURCE_BUILDERS: list[tuple[type, Callable[[object, Path], BaseDataSource]]] = [
    (BindingSitesInput, _build_binding_sites),
    (SequenceLibraryInput, _build_sequence_library),
    (PWMMemeInput, _build_pwm_meme),
    (PWMMemeSetInput, _build_pwm_meme_set),
    (PWMJasparInput, _build_pwm_jaspar),
    (PWMMatrixCSVInput, _build_pwm_matrix_csv),
    (PWMArtifactInput, _build_pwm_artifact),
    (PWMArtifactSetInput, _build_pwm_artifact_set),
    (USRSequencesInput, _build_usr_sequences),
]


def data_source_factory(cfg, cfg_path: Path) -> BaseDataSource:
    for cfg_type, builder in _SOURCE_BUILDERS:
        if isinstance(cfg, cfg_type):
            return builder(cfg, cfg_path)
    supported = ", ".join(sorted(cfg_type.__name__ for cfg_type, _ in _SOURCE_BUILDERS))
    raise ValueError(f"Unsupported source config type: {type(cfg)} (expected one of: {supported})")
