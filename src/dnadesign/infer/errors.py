"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/errors.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class InferError(Exception):
    """Base exception for this package."""


class ConfigError(InferError):
    pass


class ModelLoadError(InferError):
    pass


class ValidationError(InferError):
    pass


class CapabilityError(InferError):
    pass


class RuntimeOOMError(InferError):
    pass


class WriteBackError(InferError):
    pass
