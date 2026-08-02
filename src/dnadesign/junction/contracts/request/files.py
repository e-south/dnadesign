"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/contracts/request/files.py

Read-only JSON and YAML loading for junction requests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.tokens import AliasToken, AnchorToken

from ...errors import JunctionConfigError
from .codec import parse_request
from .limits import MAX_REQUEST_BYTES
from .model import JunctionRequest

_MAX_REQUEST_BYTES = MAX_REQUEST_BYTES


class _DuplicateJsonKeyError(ValueError):
    pass


class _DuplicateYamlKeyError(ConstructorError):
    pass


def _descriptor_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return metadata that must remain stable across one descriptor read."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json_document(source: str, request_path: Path) -> object:
    """Parse JSON while preserving the typed request-file error boundary."""

    try:
        return json.loads(source, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as exc:
        raise JunctionConfigError(f"Invalid JSON in junction request: {request_path}") from exc
    except _DuplicateJsonKeyError as exc:
        raise JunctionConfigError(f"Duplicate key in junction request: {request_path}: {exc}") from exc
    except RecursionError as exc:
        raise JunctionConfigError(f"Invalid JSON in junction request: {request_path}") from exc
    except ValueError as exc:
        raise JunctionConfigError(f"Invalid JSON in junction request: {request_path}") from exc


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping keys must be hashable",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise _DuplicateYamlKeyError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate mapping key: {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_request(path: str | Path) -> JunctionRequest:
    """Load and validate one JSON or YAML request without writing to disk."""

    request_path = Path(path).expanduser()
    if not request_path.is_absolute():
        request_path = Path.cwd() / request_path
    suffix = request_path.suffix.lower()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(request_path, flags)
    except OSError as exc:
        raise JunctionConfigError(f"Unable to open junction request: {request_path}") from exc

    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise JunctionConfigError(f"junction request is not a regular file: {request_path}")
        if opened.st_size > _MAX_REQUEST_BYTES:
            raise JunctionConfigError(
                f"junction request exceeds the {_MAX_REQUEST_BYTES}-byte input limit: {request_path}"
            )
        chunks: list[bytes] = []
        remaining = _MAX_REQUEST_BYTES
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        observed = os.fstat(descriptor)
        if len(content) > _MAX_REQUEST_BYTES or observed.st_size > _MAX_REQUEST_BYTES:
            raise JunctionConfigError(
                f"junction request exceeds the {_MAX_REQUEST_BYTES}-byte input limit: {request_path}"
            )
        if _descriptor_identity(opened) != _descriptor_identity(observed):
            raise JunctionConfigError(f"junction request changed while it was being read: {request_path}")
    except OSError as exc:
        raise JunctionConfigError(f"Unable to read junction request: {request_path}") from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass

    try:
        source = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JunctionConfigError(f"junction request is not valid UTF-8: {request_path}") from exc

    if suffix == ".json":
        payload = _load_json_document(source, request_path)
    elif suffix in {".yaml", ".yml"}:
        try:
            if any(isinstance(token, (AliasToken, AnchorToken)) for token in yaml.scan(source)):
                raise JunctionConfigError("junction YAML requests must not use anchors or aliases")
            payload = yaml.load(source, Loader=_UniqueKeySafeLoader)
        except JunctionConfigError:
            raise
        except _DuplicateYamlKeyError as exc:
            raise JunctionConfigError(f"Duplicate key in junction request: {request_path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise JunctionConfigError(f"Invalid YAML in junction request: {request_path}") from exc
        except RecursionError as exc:
            raise JunctionConfigError(f"Invalid YAML in junction request: {request_path}") from exc
        except ValueError as exc:
            raise JunctionConfigError(f"Invalid YAML in junction request: {request_path}") from exc
    else:
        raise JunctionConfigError(f"junction request must use a .json, .yaml, or .yml extension: {request_path}")

    if not isinstance(payload, dict):
        raise JunctionConfigError("junction request document must contain one object")
    return parse_request(payload)


__all__ = ["load_request"]
