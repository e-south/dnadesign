"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/contracts.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dataclasses import dataclass


class BaseRenderError(Exception): ...


class SchemaError(BaseRenderError): ...


class AlphabetError(BaseRenderError): ...


class BoundsError(BaseRenderError): ...


class RenderingError(BaseRenderError): ...


class ExportError(BaseRenderError): ...


class PluginError(BaseRenderError): ...


class SkipRecord(BaseRenderError): ...


def ensure(cond: bool, msg: str, exc: type[BaseRenderError] = BaseRenderError) -> None:
    if not cond:
        raise exc(msg)


@dataclass(frozen=True)
class Size:
    width: float
    height: float
