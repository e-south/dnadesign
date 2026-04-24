"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/datasets/lifecycle/__init__.py

Dataset lifecycle helper package for registry freeze/state and write-session
coordination.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .registry import (
    auto_freeze_registry,
    base_metadata,
    dataset_registry_hash,
    freeze_registry,
    frozen_registry_path,
    load_dataset_registry,
    require_dataset_exists,
    require_registry_for_mutation,
    stored_dataset_registry_hash,
    tombstone_path,
)
from .snapshot import snapshot_dataset
from .write_session import DatasetWriteSession, init_dataset

__all__ = [
    "DatasetWriteSession",
    "auto_freeze_registry",
    "base_metadata",
    "dataset_registry_hash",
    "freeze_registry",
    "frozen_registry_path",
    "init_dataset",
    "load_dataset_registry",
    "require_dataset_exists",
    "require_registry_for_mutation",
    "snapshot_dataset",
    "stored_dataset_registry_hash",
    "tombstone_path",
]
