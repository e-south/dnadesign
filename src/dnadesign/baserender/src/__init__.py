"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/__init__.py

Minimal biological sequence rendering.

Public API:
- SeqRecord, Annotation
- read_records, render_image, render_images, render_video

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .api import read_records, render_image, render_images, render_video
from .model import Annotation, SeqRecord

__all__ = [
    "SeqRecord",
    "Annotation",
    "read_records",
    "render_image",
    "render_images",
    "render_video",
]
__version__ = "0.1.0"
