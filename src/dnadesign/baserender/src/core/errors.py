"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/core/errors.py

Error types for baserender contract, schema, rendering, and export failures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class BaseRenderError(Exception):
    pass


class SchemaError(BaseRenderError):
    pass


class ContractError(BaseRenderError):
    pass


class AlphabetError(ContractError):
    pass


class BoundsError(ContractError):
    pass


class RenderingError(BaseRenderError):
    pass


class ExportError(BaseRenderError):
    pass


class PluginError(BaseRenderError):
    pass


class SkipRecord(BaseRenderError):
    pass
