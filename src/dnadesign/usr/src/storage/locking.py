"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/storage/locking.py

Dataset lock exports for USR write and maintenance operations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from ..locks import dataset_write_lock

__all__ = ["dataset_write_lock"]
