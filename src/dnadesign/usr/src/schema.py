"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/schema.py

Arrow schema for USR v1 required columns. These fields are immutable and
present in every dataset table.

Derived data must be attached under namespaced columns (`<tool>__<field>`).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pyarrow as pa

REQUIRED_COLUMNS = [
    ("id", pa.string()),
    ("bio_type", pa.string()),
    ("sequence", pa.string()),
    ("alphabet", pa.string()),
    ("length", pa.int32()),
    ("source", pa.string()),
    ("created_at", pa.timestamp("us", tz="UTC")),
]
ARROW_SCHEMA = pa.schema(REQUIRED_COLUMNS)
