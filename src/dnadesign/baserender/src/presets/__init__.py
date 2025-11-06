"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/presets/__init__.py

Minimal biological sequence rendering.

Public API:
- SeqRecord, Annotation
- read_records, render_image, render_images, render_video

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .loader import Job, VideoCfg, load_job, resolve_job_path

__all__ = ["Job", "VideoCfg", "load_job", "resolve_job_path"]

__version__ = "0.1.0"
