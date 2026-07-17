"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_table_schema.py

Composed ordered schemas for behavior shadow publication tables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .multistate_behavior_table_schema_completion import COMPLETION_TABLE_COLUMNS
from .multistate_behavior_table_schema_evidence import EVIDENCE_TABLE_COLUMNS

if set(EVIDENCE_TABLE_COLUMNS) & set(COMPLETION_TABLE_COLUMNS):
    raise RuntimeError("behavior evidence and completion table schema identities overlap.")

TABLE_COLUMNS = {**EVIDENCE_TABLE_COLUMNS, **COMPLETION_TABLE_COLUMNS}


__all__ = ["TABLE_COLUMNS"]
