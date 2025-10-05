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


class ConfigError(InferError): ...


class ModelLoadError(InferError): ...


class ValidationError(InferError): ...


class CapabilityError(InferError): ...


class RuntimeOOMError(InferError): ...


class WriteBackError(InferError): ...


class IOErrorInfer(InferError):
    """File I/O related issues."""


class UnsafeInputError(InferError):
    """Raised when attempting unsafe operations (e.g., pickle load) without consent."""
