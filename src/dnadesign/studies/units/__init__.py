"""
Concrete dnadesign study source units.

Shared study infrastructure lives in ``dnadesign.studies.core``. Concrete
study-specific compilers, status services, preflights, and handoff helpers live
under this package so they do not sprawl beside package assets or shared core
code. Tests for each concrete study live under the owning
``dnadesign.studies.units.<study_id>.tests`` package.
"""

from __future__ import annotations

__all__ = []
