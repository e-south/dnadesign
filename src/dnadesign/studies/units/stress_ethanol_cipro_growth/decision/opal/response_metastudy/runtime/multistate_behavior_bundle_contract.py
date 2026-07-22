"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/multistate_behavior_bundle_contract.py

Persisted identities and required fields for behavior shadow bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .multistate_behavior_table_schema import TABLE_COLUMNS

SCHEMA_ID = "stress_ethanol_cipro_growth.multistate_response_behavior_shadow_bundle.v1"
TABLE_IDS = frozenset(TABLE_COLUMNS)
REQUIRED_COLUMNS = {table_id: frozenset(columns) for table_id, columns in TABLE_COLUMNS.items()}

__all__ = ["REQUIRED_COLUMNS", "SCHEMA_ID", "TABLE_COLUMNS", "TABLE_IDS"]
