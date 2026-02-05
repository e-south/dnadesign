"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/schema.py

USR schema definitions and metadata keys.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pyarrow as pa

SCHEMA_VERSION = "1"
ID_HASH_SPEC = "sha1_utf8_pipe"
META_SCHEMA_VERSION = "usr:schema_version"
META_DATASET_CREATED_AT = "usr:dataset_created_at"
META_ID_HASH = "usr:id_hash"

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


def base_metadata(created_at: str | None) -> dict[bytes, bytes]:
    md = {
        META_SCHEMA_VERSION.encode("utf-8"): SCHEMA_VERSION.encode("utf-8"),
        META_ID_HASH.encode("utf-8"): ID_HASH_SPEC.encode("utf-8"),
    }
    if created_at:
        md[META_DATASET_CREATED_AT.encode("utf-8")] = str(created_at).encode("utf-8")
    return md


def merge_base_metadata(existing: dict[bytes, bytes] | None, created_at: str | None = None) -> dict[bytes, bytes]:
    md = dict(existing or {})
    if META_SCHEMA_VERSION.encode("utf-8") not in md:
        md[META_SCHEMA_VERSION.encode("utf-8")] = SCHEMA_VERSION.encode("utf-8")
    if META_ID_HASH.encode("utf-8") not in md:
        md[META_ID_HASH.encode("utf-8")] = ID_HASH_SPEC.encode("utf-8")
    if created_at and META_DATASET_CREATED_AT.encode("utf-8") not in md:
        md[META_DATASET_CREATED_AT.encode("utf-8")] = str(created_at).encode("utf-8")
    return md


def with_base_metadata(
    table: pa.Table,
    *,
    created_at: str | None = None,
    existing: dict[bytes, bytes] | None = None,
) -> pa.Table:
    md = merge_base_metadata(existing, created_at)
    return table.replace_schema_metadata(md)
