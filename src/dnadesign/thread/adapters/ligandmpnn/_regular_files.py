"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/_regular_files.py

Descriptor-bound regular-file reads for LigandMPNN inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


class NonRegularFileError(OSError):
    """The opened leaf is not a regular file."""


@contextmanager
def open_regular_file(path: Path) -> Iterator[BinaryIO]:
    """Open one no-follow, nonblocking leaf and verify that exact descriptor."""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        nonregular_errnos = {errno.ELOOP}
        if hasattr(errno, "EFTYPE"):
            nonregular_errnos.add(errno.EFTYPE)
        if error.errno in nonregular_errnos:
            raise NonRegularFileError(errno.EINVAL, "path is not a regular file", path) from error
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise NonRegularFileError(errno.EINVAL, "path is not a regular file", path)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle
    finally:
        os.close(descriptor)
