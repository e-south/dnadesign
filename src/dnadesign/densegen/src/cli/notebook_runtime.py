"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/src/cli/notebook_runtime.py

Runtime helpers for DenseGen notebook launch behavior.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import html
import ipaddress
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

PORT_DISCOVERY_ATTEMPTS = 32
BROWSER_READY_TIMEOUT_SECONDS = 30.0
WILDCARD_BIND_HOSTS = {"0.0.0.0", "::"}


def resolve_tcp_bind_targets(
    host: str,
    port: int,
    *,
    socket_module=socket,
) -> list[tuple[int, int, int, tuple[object, ...]]]:
    try:
        raw_targets = socket_module.getaddrinfo(
            host,
            port,
            family=socket_module.AF_UNSPEC,
            type=socket_module.SOCK_STREAM,
            proto=socket_module.IPPROTO_TCP,
        )
    except OSError:
        return []
    targets: list[tuple[int, int, int, tuple[object, ...]]] = []
    seen: set[tuple[int, int, int, tuple[object, ...]]] = set()
    for family, socktype, proto, _, sockaddr in raw_targets:
        if not isinstance(sockaddr, tuple):
            continue
        normalized = tuple(sockaddr)
        target = (int(family), int(socktype), int(proto), normalized)
        if target in seen:
            continue
        seen.add(target)
        targets.append(target)
    return targets


def port_is_available(
    host: str,
    port: int,
    *,
    socket_module=socket,
) -> bool:
    targets = resolve_tcp_bind_targets(host, port, socket_module=socket_module)
    if not targets:
        return False
    for family, socktype, proto, sockaddr in targets:
        try:
            with socket_module.socket(family, socktype, proto) as sock:
                sock.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
                sock.bind(sockaddr)
        except OSError:
            return False
    return True


def find_available_port(
    host: str,
    *,
    socket_module=socket,
    port_discovery_attempts: int = PORT_DISCOVERY_ATTEMPTS,
    port_is_available_fn: Callable[[str, int], bool] = port_is_available,
) -> int | None:
    targets = resolve_tcp_bind_targets(host, 0, socket_module=socket_module)
    if not targets:
        return None
    family, socktype, proto, sockaddr = targets[0]
    for _ in range(int(port_discovery_attempts)):
        try:
            with socket_module.socket(family, socktype, proto) as sock:
                sock.setsockopt(socket_module.SOL_SOCKET, socket_module.SO_REUSEADDR, 1)
                sock.bind(sockaddr)
                candidate_port = int(sock.getsockname()[1])
        except OSError:
            return None
        if candidate_port <= 0:
            continue
        if port_is_available_fn(host, candidate_port):
            return candidate_port
    return None


def url_is_reachable(
    url: str,
    *,
    urllib_request_module=urllib.request,
    urllib_error_module=urllib.error,
) -> bool:
    request = urllib_request_module.Request(url, method="GET")
    try:
        with urllib_request_module.urlopen(request, timeout=0.5) as response:
            content_type = str(response.headers.get("content-type", "")).lower()
            body = response.read(2048).decode("utf-8", errors="ignore").lower()
            if int(getattr(response, "status", 200)) >= 400:
                return False
            if "text/html" not in content_type:
                return False
            if 'data-marimo="true"' in body or "data-marimo='true'" in body:
                return True
            marimo_markers = ("a marimo app", "marimo", "favicon.ico")
            return all(marker in body for marker in marimo_markers)
    except (OSError, urllib_error_module.URLError, TimeoutError, ValueError):
        return False


def running_notebook_filename(
    url: str,
    *,
    urllib_request_module=urllib.request,
    urllib_error_module=urllib.error,
) -> str | None:
    request = urllib_request_module.Request(url, method="GET")
    try:
        with urllib_request_module.urlopen(request, timeout=0.5) as response:
            content_type = str(response.headers.get("content-type", "")).lower()
            if int(getattr(response, "status", 200)) >= 400:
                return None
            if "text/html" not in content_type:
                return None
            body = response.read(32768).decode("utf-8", errors="ignore")
    except (OSError, urllib_error_module.URLError, TimeoutError, ValueError):
        return None
    match = re.search(r"<marimo-filename[^>]*>(.*?)</marimo-filename>", body, re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    raw = str(match.group(1) or "").strip()
    if not raw:
        return None
    return html.unescape(raw)


def resolve_browser_host(host: str) -> str:
    host_value = str(host).strip()
    return "localhost" if host_value in WILDCARD_BIND_HOSTS else host_value


def format_http_url(host: str, port: int, *, for_browser: bool) -> str:
    url_host = resolve_browser_host(host) if for_browser else str(host).strip()
    try:
        parsed = ipaddress.ip_address(url_host)
    except ValueError:
        return f"http://{url_host}:{port}"
    if parsed.version == 6:
        return f"http://[{url_host}]:{port}"
    return f"http://{url_host}:{port}"


def is_wsl(
    *,
    os_module=os,
    sys_module=sys,
    path_cls=Path,
) -> bool:
    if not (os_module.name == "posix" and sys_module.platform.startswith("linux")):
        return False
    release_paths = ("/proc/sys/kernel/osrelease", "/proc/version")
    for raw_path in release_paths:
        path = path_cls(raw_path)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if "microsoft" in text or "wsl" in text:
            return True
    return False


def open_browser_tab(
    url: str,
    *,
    is_wsl_fn: Callable[[], bool] = is_wsl,
    shutil_module=shutil,
    sys_module=sys,
    os_module=os,
    subprocess_module=subprocess,
) -> bool:
    command_groups: list[list[str]] = []
    if is_wsl_fn():
        if shutil_module.which("wslview"):
            command_groups.append(["wslview", url])
        if shutil_module.which("powershell.exe"):
            command_groups.append(["powershell.exe", "-NoProfile", "-Command", "Start-Process", url])
        if shutil_module.which("cmd.exe"):
            command_groups.append(["cmd.exe", "/c", "start", "", url])
    elif sys_module.platform == "darwin":
        if shutil_module.which("open"):
            command_groups.append(["open", url])
    elif os_module.name == "nt":
        if shutil_module.which("powershell"):
            command_groups.append(["powershell", "-NoProfile", "-Command", "Start-Process", url])
        if shutil_module.which("cmd"):
            command_groups.append(["cmd", "/c", "start", "", url])
    else:
        if shutil_module.which("xdg-open"):
            command_groups.append(["xdg-open", url])

    for command in command_groups:
        try:
            subprocess_module.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            continue
    return False


def process_is_running(
    pid: int,
    *,
    os_module=os,
) -> bool:
    pid_value = int(pid)
    if pid_value <= 0:
        return False
    try:
        os_module.kill(pid_value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def terminate_process_tree(
    pid: int,
    *,
    timeout_seconds: float = 3.0,
    os_module=os,
    signal_module=signal,
    time_module=time,
) -> bool:
    pid_value = int(pid)
    if pid_value <= 0:
        return False
    if not process_is_running(pid_value, os_module=os_module):
        return True

    def _is_alive() -> bool:
        return process_is_running(pid_value, os_module=os_module)

    if os_module.name == "posix":
        try:
            pgid = os_module.getpgid(pid_value)
        except OSError:
            pgid = None
        try:
            if pgid is not None:
                os_module.killpg(pgid, signal_module.SIGTERM)
            else:
                os_module.kill(pid_value, signal_module.SIGTERM)
        except ProcessLookupError:
            return True
        except OSError:
            return False
    else:
        try:
            os_module.kill(pid_value, signal_module.SIGTERM)
        except ProcessLookupError:
            return True
        except OSError:
            return False

    deadline = time_module.monotonic() + float(timeout_seconds)
    while time_module.monotonic() < deadline:
        if not _is_alive():
            return True
        time_module.sleep(0.05)

    if os_module.name == "posix":
        try:
            if pgid is not None:
                os_module.killpg(pgid, signal_module.SIGKILL)
            else:
                os_module.kill(pid_value, signal_module.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            return False
    else:
        try:
            os_module.kill(pid_value, signal_module.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            return False

    kill_deadline = time_module.monotonic() + float(timeout_seconds)
    while time_module.monotonic() < kill_deadline:
        if not _is_alive():
            return True
        time_module.sleep(0.05)
    return not _is_alive()


def run_marimo_command(
    *,
    command: list[str],
    env: dict[str, str],
    browser_url: str | None = None,
    open_timeout_seconds: float = BROWSER_READY_TIMEOUT_SECONDS,
    on_browser_open_failure: Callable[[str], None] | None = None,
    on_process_start: Callable[[int], None] | None = None,
    subprocess_module=subprocess,
    signal_module=signal,
    os_module=os,
    time_module=time,
    url_is_reachable_fn: Callable[[str], bool] = url_is_reachable,
    open_browser_tab_fn: Callable[[str], bool] = open_browser_tab,
) -> bool:
    process = subprocess_module.Popen(command, env=env, start_new_session=(os_module.name == "posix"))
    if on_process_start is not None:
        try:
            on_process_start(int(process.pid))
        except Exception:
            pass
    opened = False
    warned = False
    original_signal_handlers: dict[int, object] = {}

    def _terminate_running_process(force: bool = False) -> None:
        if process.poll() is not None:
            return
        if os_module.name == "posix":
            try:
                sig = signal_module.SIGKILL if force else signal_module.SIGTERM
                os_module.killpg(os_module.getpgid(process.pid), sig)
                return
            except Exception:
                pass
        try:
            if force:
                process.kill()
            else:
                process.terminate()
        except Exception:
            pass

    def _on_signal(signum: int, _frame) -> None:
        _terminate_running_process(force=False)
        if signum == getattr(signal_module, "SIGINT", None):
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))

    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if not hasattr(signal_module, signal_name):
            continue
        sig = getattr(signal_module, signal_name)
        try:
            original_signal_handlers[int(sig)] = signal_module.getsignal(sig)
            signal_module.signal(sig, _on_signal)
        except Exception:
            continue

    try:
        if browser_url is None:
            return_code = process.wait()
            if return_code != 0:
                raise subprocess_module.CalledProcessError(return_code, command)
            return False

        deadline = time_module.monotonic() + float(open_timeout_seconds)
        while time_module.monotonic() < deadline:
            if process.poll() is not None:
                break
            if url_is_reachable_fn(browser_url):
                opened = open_browser_tab_fn(browser_url)
                if not opened and on_browser_open_failure is not None:
                    on_browser_open_failure("browser-open-failed")
                    warned = True
                break
            time_module.sleep(0.2)
        if not opened and not warned and process.poll() is None and on_browser_open_failure is not None:
            on_browser_open_failure("notebook-not-reachable")
        return_code = process.wait()
        if return_code != 0:
            raise subprocess_module.CalledProcessError(return_code, command)
        return opened
    except BaseException:
        if process.poll() is None:
            _terminate_running_process(force=False)
            try:
                process.wait(timeout=3.0)
            except Exception:
                _terminate_running_process(force=True)
        raise
    finally:
        for sig_num, handler in original_signal_handlers.items():
            try:
                signal_module.signal(sig_num, handler)
            except Exception:
                continue
