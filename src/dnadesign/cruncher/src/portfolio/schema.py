"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/portfolio/schema.py

Schema registry and validation entrypoint for Portfolio specs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TypeAlias

from dnadesign.cruncher.portfolio.schema_models import PortfolioRoot, PortfolioSpec

PortfolioSpecModel: TypeAlias = PortfolioSpec
PortfolioRootModel: TypeAlias = PortfolioRoot

PORTFOLIO_SCHEMA_ROOT_BY_VERSION: dict[int, type[PortfolioRootModel]] = {
    3: PortfolioRoot,
}


def parse_portfolio_root(payload: dict[str, object]) -> PortfolioRootModel:
    portfolio_payload = payload.get("portfolio")
    if not isinstance(portfolio_payload, dict):
        raise ValueError("Portfolio schema required (portfolio must be a mapping)")
    version = portfolio_payload.get("schema_version")
    root_model = PORTFOLIO_SCHEMA_ROOT_BY_VERSION.get(version) if isinstance(version, int) else None
    if root_model is None:
        root_model = PortfolioRoot
    return root_model.model_validate(payload)
