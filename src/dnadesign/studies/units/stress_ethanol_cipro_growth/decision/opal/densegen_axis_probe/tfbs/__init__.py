"""DenseGen TFBS learnability probe contracts and execution surfaces.

This subpackage owns the strict TFBS construction-label probe. The package root
keeps the older DenseGen plan-logic probe; TFBS-specific parsing, nulls, Stage A
materialization, Stage B execution, and realized-label review stay here.

Import concrete modules directly. The subpackage root intentionally exports
nothing so ownership boundaries remain explicit.
"""

from __future__ import annotations

__all__: list[str] = []
