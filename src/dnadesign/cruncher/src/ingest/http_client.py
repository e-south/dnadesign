"""HTTP helpers with retry/backoff for ingestion adapters."""

from __future__ import annotations

import json
import random
import time
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError


@dataclass(frozen=True)
class HttpRetryPolicy:
    retries: int = 3
    backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)
    respect_retry_after: bool = True

    def sleep_for(self, attempt: int, retry_after: Optional[float]) -> None:
        if retry_after is not None and self.respect_retry_after:
            delay = min(self.max_backoff_seconds, retry_after)
        else:
            delay = min(self.max_backoff_seconds, self.backoff_seconds * (2**attempt))
        delay = delay + random.uniform(0, max(0.0, delay * 0.25))
        time.sleep(delay)


def request_bytes(
    url: str,
    *,
    data: Optional[bytes] = None,
    headers: Optional[dict[str, str]] = None,
    method: Optional[str] = None,
    timeout: int = 30,
    context: Optional[object] = None,
    retry: Optional[HttpRetryPolicy] = None,
) -> bytes:
    policy = retry or HttpRetryPolicy()
    headers = headers or {}
    last_exc: Optional[Exception] = None
    for attempt in range(policy.retries + 1):
        try:
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=timeout, context=context) as resp:
                return resp.read()
        except HTTPError as exc:
            last_exc = exc
            status = exc.code
            retry_after = _parse_retry_after(exc.headers)
            if status in policy.retry_statuses and attempt < policy.retries:
                policy.sleep_for(attempt, retry_after)
                continue
            raise
        except URLError as exc:
            last_exc = exc
            if attempt < policy.retries:
                policy.sleep_for(attempt, None)
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("request failed without exception")


def request_json(
    url: str,
    *,
    data: Optional[bytes] = None,
    headers: Optional[dict[str, str]] = None,
    method: Optional[str] = None,
    timeout: int = 30,
    context: Optional[object] = None,
    retry: Optional[HttpRetryPolicy] = None,
) -> dict:
    payload = request_bytes(
        url,
        data=data,
        headers=headers,
        method=method,
        timeout=timeout,
        context=context,
        retry=retry,
    )
    return json.loads(payload.decode("utf-8"))


def download_to(
    url: str,
    dest_path: str,
    *,
    timeout: int = 30,
    context: Optional[object] = None,
    retry: Optional[HttpRetryPolicy] = None,
) -> None:
    payload = request_bytes(
        url,
        timeout=timeout,
        context=context,
        retry=retry,
    )
    with open(dest_path, "wb") as fh:
        fh.write(payload)


def _parse_retry_after(headers: Optional[Iterable[str]]) -> Optional[float]:
    if not headers:
        return None
    try:
        value = headers.get("Retry-After") if hasattr(headers, "get") else None
    except Exception:
        value = None
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
