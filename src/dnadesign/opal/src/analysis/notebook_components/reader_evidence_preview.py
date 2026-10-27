"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_evidence_preview.py

Image previews for Reader evidence artifacts in generated OPAL notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

READER_EVIDENCE_PDF_PREVIEW_DPI = 180
READER_EVIDENCE_PDF_PREVIEW_TIMEOUT_SECONDS = 30


def reader_pdf_preview_path(path: str | Path) -> Path:
    """Render the first PDF page to a cached PNG preview and return its path."""

    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise RuntimeError(f"Reader PDF artifact does not exist: {source_path}")
    source_stat = source_path.stat()
    cache_root = Path(tempfile.gettempdir()) / "dnadesign-opal-reader-evidence-previews"
    cache_root.mkdir(parents=True, exist_ok=True)
    digest_source = f"{source_path.resolve()}:{source_stat.st_mtime_ns}:{source_stat.st_size}".encode()
    digest = hashlib.sha256(digest_source).hexdigest()[:16]
    preview_path = cache_root / f"{source_path.stem}-{digest}.png"
    if preview_path.exists() and preview_path.stat().st_size > 0:
        return preview_path
    _render_pdf_preview(source_path=source_path, preview_path=preview_path)
    return preview_path


def _render_pdf_preview(*, source_path: Path, preview_path: Path) -> None:
    errors: list[str] = []
    ghostscript_path = shutil.which("gs")
    if ghostscript_path:
        error = _run_pdf_preview_command(
            [
                ghostscript_path,
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-dFirstPage=1",
                "-dLastPage=1",
                "-sDEVICE=pngalpha",
                f"-r{READER_EVIDENCE_PDF_PREVIEW_DPI}",
                "-o",
                str(preview_path),
                str(source_path),
            ],
            preview_path=preview_path,
        )
        if error is None:
            return
        errors.append(error)
    sips_path = shutil.which("sips")
    if sips_path:
        error = _run_pdf_preview_command(
            [sips_path, "-s", "format", "png", str(source_path), "--out", str(preview_path)],
            preview_path=preview_path,
        )
        if error is None:
            return
        errors.append(error)
    if not errors:
        raise RuntimeError("Reader PDF image previews require Ghostscript (`gs`) or macOS `sips`.")
    raise RuntimeError("Reader PDF preview rendering failed: " + "; ".join(errors))


def _run_pdf_preview_command(command: list[str], *, preview_path: Path) -> str | None:
    if preview_path.exists():
        preview_path.unlink()
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=READER_EVIDENCE_PDF_PREVIEW_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        if preview_path.exists():
            preview_path.unlink()
        return f"{Path(command[0]).name} timed out after {READER_EVIDENCE_PDF_PREVIEW_TIMEOUT_SECONDS} seconds"
    if result.returncode == 0 and preview_path.exists() and preview_path.stat().st_size > 0:
        return None
    message = "\n".join(part.strip() for part in (result.stderr, result.stdout) if part.strip())
    if not message:
        message = "no diagnostic output"
    return f"{Path(command[0]).name} exited {result.returncode}: {message}"


__all__ = [
    "READER_EVIDENCE_PDF_PREVIEW_DPI",
    "READER_EVIDENCE_PDF_PREVIEW_TIMEOUT_SECONDS",
    "reader_pdf_preview_path",
]
