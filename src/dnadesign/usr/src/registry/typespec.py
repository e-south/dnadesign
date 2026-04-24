"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/registry/typespec.py

Registry type parsing and Arrow conversion helpers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pyarrow as pa

from ..contracts import SchemaError


def parse_type_str(type_str: str) -> str:
    type_str = type_str.strip()
    if type_str in {
        "string",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "bool",
    }:
        return type_str
    if type_str.startswith("fixed_size_list<"):
        inner, size = _parse_fixed_size_list(type_str)
        return f"fixed_size_list<{parse_type_str(inner)}>[{size}]"
    if type_str.startswith("list<") and type_str.endswith(">"):
        inner = type_str[len("list<") : -1].strip()
        return f"list<{parse_type_str(inner)}>"
    if type_str.startswith("struct<") and type_str.endswith(">"):
        inner = type_str[len("struct<") : -1].strip()
        fields = _split_top_level(inner)
        if not fields:
            raise SchemaError("Struct type must include at least one field.")
        parsed_fields = []
        for field in fields:
            if ":" not in field:
                raise SchemaError(f"Struct field '{field}' must be name:type.")
            name, inner_type = field.split(":", 1)
            name = name.strip()
            inner_type = inner_type.strip()
            if not name or not inner_type:
                raise SchemaError(f"Struct field '{field}' must be name:type.")
            parsed_fields.append(f"{name}:{parse_type_str(inner_type)}")
        return f"struct<{','.join(parsed_fields)}>"
    if type_str.startswith("timestamp[") and type_str.endswith("]"):
        return type_str
    raise SchemaError(f"Unsupported registry type '{type_str}'.")


def arrow_type_str(dtype: pa.DataType) -> str:
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "string"
    if pa.types.is_int8(dtype):
        return "int8"
    if pa.types.is_int16(dtype):
        return "int16"
    if pa.types.is_int32(dtype):
        return "int32"
    if pa.types.is_int64(dtype):
        return "int64"
    if pa.types.is_uint8(dtype):
        return "uint8"
    if pa.types.is_uint16(dtype):
        return "uint16"
    if pa.types.is_uint32(dtype):
        return "uint32"
    if pa.types.is_uint64(dtype):
        return "uint64"
    if pa.types.is_float16(dtype):
        return "float16"
    if pa.types.is_float32(dtype):
        return "float32"
    if pa.types.is_float64(dtype):
        return "float64"
    if pa.types.is_boolean(dtype):
        return "bool"
    if pa.types.is_timestamp(dtype):
        tz = dtype.tz
        unit = dtype.unit
        if tz:
            return f"timestamp[{unit}, {tz}]"
        return f"timestamp[{unit}]"
    if pa.types.is_fixed_size_list(dtype):
        return f"fixed_size_list<{arrow_type_str(dtype.value_type)}>[{dtype.list_size}]"
    if pa.types.is_struct(dtype):
        fields = ",".join(f"{field.name}:{arrow_type_str(field.type)}" for field in dtype)
        return f"struct<{fields}>"
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return f"list<{arrow_type_str(dtype.value_type)}>"
    raise SchemaError(f"Unsupported Arrow type '{dtype}'.")


def _split_top_level(spec: str) -> list[str]:
    parts: list[str] = []
    if not spec:
        return parts
    depth = 0
    start = 0
    for index, ch in enumerate(spec):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            part = spec[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = spec[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_fixed_size_list(type_str: str) -> tuple[str, int]:
    prefix = "fixed_size_list<"
    if not type_str.startswith(prefix):
        raise SchemaError(f"Invalid fixed_size_list type '{type_str}'.")
    inner_spec = type_str[len(prefix) :]
    depth = 0
    inner_end = None
    for index, ch in enumerate(inner_spec):
        if ch == "<":
            depth += 1
        elif ch == ">":
            if depth == 0:
                inner_end = index
                break
            depth -= 1
    if inner_end is None:
        raise SchemaError(f"Invalid fixed_size_list type '{type_str}'.")
    inner = inner_spec[:inner_end].strip()
    rest = inner_spec[inner_end + 1 :].strip()
    if not rest.startswith("[") or not rest.endswith("]"):
        raise SchemaError(f"Invalid fixed_size_list size in '{type_str}'.")
    size_str = rest[1:-1].strip()
    if not size_str.isdigit():
        raise SchemaError(f"Invalid fixed_size_list size in '{type_str}'.")
    size = int(size_str)
    if size <= 0:
        raise SchemaError(f"fixed_size_list size must be positive in '{type_str}'.")
    return inner, size
