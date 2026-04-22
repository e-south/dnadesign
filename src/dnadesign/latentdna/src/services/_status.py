"""
Status merge helpers for latentdna command and artifact contracts.
"""

from __future__ import annotations


def merge_statuses(*statuses: str) -> str:
    normalized = [str(status).strip().lower() for status in statuses if str(status).strip()]
    if any(status == "error" for status in normalized):
        return "error"
    if any(status == "attention" for status in normalized):
        return "attention"
    if any(status == "missing" for status in normalized):
        return "missing"
    return "ok"
