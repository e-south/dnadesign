"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/contracts/base.py

Shared configuration model base for construct contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pydantic import BaseModel


class StrictConfigModel(BaseModel):
    model_config = {"extra": "forbid"}
