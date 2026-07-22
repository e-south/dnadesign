#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

FORBIDDEN_SNIPPETS = (
    ";",
    "\n",
    "\r",
    "://",
    "runscript",
    "shell",
    "python ",
)


POSE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,79}$")
INT_RE = re.compile(r"^-?[0-9]+$")
FLOAT_RE = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?$")
SAFE_SELECTION_CHARS_RE = re.compile(r"^[A-Za-z0-9#/:@&.,_+*?\-\s]+$")
SAFE_COLOR_RE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9 -]{0,40}|#[0-9A-Fa-f]{6})$")
COLOR_TARGET_CHARS = frozenset("abcspflr")
EXECUTABLE_OPEN_SUFFIXES = {".cxc", ".cmd", ".py", ".pyc", ".sh", ".bash", ".zsh", ".command"}
ALLOWED_OPEN_SUFFIXES = {
    ".bild",
    ".cif",
    ".cxs",
    ".dx",
    ".map",
    ".mae",
    ".mol2",
    ".mrc",
    ".mtz",
    ".pdb",
    ".pdbqt",
    ".sdf",
    ".xyz",
}


def _safe_selection(text: str) -> bool:
    return bool(text and SAFE_SELECTION_CHARS_RE.fullmatch(text.strip()))


def _is_local_existing_path(path_text: str) -> bool:
    path = Path(path_text).expanduser()
    return path.exists() and "://" not in path_text


def _is_allowed_open_path(path_text: str) -> bool:
    if not _is_local_existing_path(path_text):
        return False
    suffix = Path(path_text).suffix.lower()
    return suffix in ALLOWED_OPEN_SUFFIXES and suffix not in EXECUTABLE_OPEN_SUFFIXES


def _read_simple_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def _resolve_port(*, port: int | None, session_manifest: Path | None) -> int:
    if port is not None:
        return port
    if session_manifest is None:
        raise ValueError("provide --port or --session-manifest")
    values = _read_simple_manifest(session_manifest)
    raw_port = values.get("port")
    if raw_port is None:
        raise ValueError(f"session manifest has no port: {session_manifest}")
    return int(raw_port)


def _session_command_log_path(session_manifest: Path | None) -> Path | None:
    if session_manifest is None:
        return None
    values = _read_simple_manifest(session_manifest)
    raw_path = values.get("command_log_path")
    if not raw_path:
        return None
    return Path(raw_path)


def _append_command_log(
    *,
    command_log_path: Path | None,
    port: int,
    command: str,
    status: str,
    response: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    if command_log_path is None:
        return
    command_log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "port": port,
        "command": command,
        "status": status,
        "response": response,
        "error": error,
    }
    with command_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _allowed_save(parts: list[str]) -> bool:
    if len(parts) == 2 and parts[1].endswith(".cxs"):
        return True
    if len(parts) == 8 and parts[1].endswith(".png"):
        return (
            parts[2] == "width"
            and parts[4] == "height"
            and parts[6] == "supersample"
            and INT_RE.fullmatch(parts[3]) is not None
            and INT_RE.fullmatch(parts[5]) is not None
            and INT_RE.fullmatch(parts[7]) is not None
        )
    return False


def _allowed_show_hide(parts: list[str]) -> bool:
    if len(parts) < 3:
        return False
    if parts[-2] == "target":
        return parts[-1] in {"a", "b", "c", "r", "s", "m", "acs", "ac", "rs"} and _safe_selection(" ".join(parts[1:-2]))
    return parts[-1] in {"atoms", "cartoons", "surfaces", "models", "pseudobonds"} and _safe_selection(
        " ".join(parts[1:-1])
    )


def _allowed_nucleotides(parts: list[str]) -> bool:
    return len(parts) >= 3 and parts[-1] in {"atoms", "ladder"} and _safe_selection(" ".join(parts[1:-1]))


def _allowed_cartoon(parts: list[str]) -> bool:
    if len(parts) == 2:
        return _safe_selection(parts[1])
    if len(parts) == 4 and parts[2] == "suppressBackboneDisplay" and parts[3] in {"true", "false"}:
        return _safe_selection(parts[1])
    if len(parts) == 6 and parts[:3] == ["cartoon", "style", "width"] and parts[4] == "thick":
        return FLOAT_RE.fullmatch(parts[3]) is not None and FLOAT_RE.fullmatch(parts[5]) is not None
    if (
        len(parts) == 9
        and parts[:4] == ["cartoon", "style", "nucleic", "xsect"]
        and parts[4] in {"oval", "rectangle", "barbell"}
        and parts[5] == "width"
        and parts[7] == "thick"
    ):
        return FLOAT_RE.fullmatch(parts[6]) is not None and FLOAT_RE.fullmatch(parts[8]) is not None
    if (
        len(parts) == 11
        and parts[:3] == ["cartoon", "tether", "nucleic"]
        and parts[3:5] == ["shape", "cylinder"]
        and parts[5] == "sides"
        and parts[7] == "scale"
        and parts[9] == "opacity"
    ):
        return (
            INT_RE.fullmatch(parts[6]) is not None
            and FLOAT_RE.fullmatch(parts[8]) is not None
            and FLOAT_RE.fullmatch(parts[10]) is not None
        )
    return False


def _allowed_size(parts: list[str]) -> bool:
    return (
        len(parts) == 4
        and parts[2] == "stickRadius"
        and _safe_selection(parts[1])
        and FLOAT_RE.fullmatch(parts[3]) is not None
    )


def _allowed_shape_ribbon(parts: list[str]) -> bool:
    if len(parts) != 13 or parts[:2] != ["shape", "ribbon"]:
        return False
    return (
        _safe_selection(parts[2])
        and parts[3] == "width"
        and FLOAT_RE.fullmatch(parts[4]) is not None
        and parts[5] == "height"
        and FLOAT_RE.fullmatch(parts[6]) is not None
        and parts[7:9] == ["followBonds", "false"]
        and parts[9] == "color"
        and SAFE_COLOR_RE.fullmatch(parts[10]) is not None
        and parts[11] == "modelId"
        and re.fullmatch(r"#[0-9]+", parts[12]) is not None
    )


def _allowed_name(parts: list[str]) -> bool:
    return len(parts) >= 3 and POSE_ID_RE.fullmatch(parts[1]) is not None and _safe_selection(" ".join(parts[2:]))


def _allowed_rename(parts: list[str]) -> bool:
    return len(parts) == 3 and _safe_selection(parts[1]) and POSE_ID_RE.fullmatch(parts[2]) is not None


def _allowed_view(parts: list[str]) -> bool:
    if len(parts) == 3 and parts[:2] == ["view", "name"]:
        return POSE_ID_RE.fullmatch(parts[2]) is not None
    if len(parts) == 2:
        return POSE_ID_RE.fullmatch(parts[1]) is not None or parts[1] in {"all", "initial"}
    if len(parts) == 4 and parts[2] == "pad":
        return (parts[1] == "all" or _safe_selection(parts[1])) and FLOAT_RE.fullmatch(parts[3]) is not None
    return False


def _allowed_color(parts: list[str]) -> bool:
    if len(parts) < 5 or parts[-2] != "target":
        return False
    target = parts[-1]
    color = parts[-3]
    selection = " ".join(parts[1:-3])
    return (
        bool(target)
        and len(target) <= len(COLOR_TARGET_CHARS)
        and set(target) <= COLOR_TARGET_CHARS
        and _safe_selection(selection)
        and bool(SAFE_COLOR_RE.fullmatch(color))
    )


def _allowed_surface(parts: list[str]) -> bool:
    if len(parts) < 2:
        return False
    if parts[1] in {"close", "hidePatches", "showPatches"}:
        return len(parts) >= 3 and _safe_selection(" ".join(parts[2:]))
    if len(parts) == 2:
        return _safe_selection(parts[1])
    if len(parts) == 6 and parts[2] == "color" and parts[4] == "transparency":
        return _safe_selection(parts[1]) and bool(SAFE_COLOR_RE.fullmatch(parts[3])) and INT_RE.fullmatch(parts[5])
    return False


def _allowed_transparency(parts: list[str]) -> bool:
    if len(parts) < 5 or parts[-2] != "target":
        return False
    return (
        INT_RE.fullmatch(parts[-3]) is not None
        and parts[-1] in {"a", "c", "r", "s", "ac", "rs"}
        and _safe_selection(" ".join(parts[1:-3]))
    )


def _allowed_title(command: str, parts: list[str]) -> bool:
    if command == "2dlabels delete all":
        return True
    if len(parts) < 13 or parts[0:2] != ["2dlabels", "text"]:
        return False
    return "xpos" in parts and "ypos" in parts and "size" in parts and "color" in parts and "bgColor" in parts


def _allowed(command: str) -> bool:
    stripped = command.strip()
    lowered = stripped.lower()
    if any(snippet in lowered for snippet in FORBIDDEN_SNIPPETS):
        return False
    try:
        parts = shlex.split(stripped)
    except ValueError:
        return False
    if not parts:
        return False
    if parts == ["remotecontrol", "rest", "port"]:
        return True
    if parts == ["remotecontrol", "rest", "stop"]:
        return True
    if parts[:2] == ["set", "bgColor"] and len(parts) == 3:
        return bool(SAFE_COLOR_RE.fullmatch(parts[2]))
    if parts == ["camera", "ortho"] or parts == ["camera", "mono"]:
        return True
    if parts[0] == "view":
        return _allowed_view(parts)
    if parts[0] == "save":
        return _allowed_save(parts)
    if parts[0] == "open" and len(parts) == 2:
        return _is_allowed_open_path(parts[1])
    if parts == ["close", "session"]:
        return True
    if parts == ["label", "delete"]:
        return True
    if parts[0] in {"show", "hide"}:
        return _allowed_show_hide(parts)
    if parts[0] == "nucleotides":
        return _allowed_nucleotides(parts)
    if parts[0] == "name":
        return _allowed_name(parts)
    if parts[0] == "rename":
        return _allowed_rename(parts)
    if parts[0] == "style" and len(parts) >= 3:
        return parts[-1] in {"stick", "ball", "sphere"} and _safe_selection(" ".join(parts[1:-1]))
    if parts[0] == "size":
        return _allowed_size(parts)
    if parts[0] == "surface":
        return _allowed_surface(parts)
    if parts[0] == "transparency":
        return _allowed_transparency(parts)
    if parts[0] == "color":
        return _allowed_color(parts)
    if parts[0] == "cartoon":
        return _allowed_cartoon(parts)
    if parts[0] == "shape":
        return _allowed_shape_ribbon(parts)
    if parts == ["lighting", "soft"] or parts == ["lighting", "full"]:
        return True
    if parts == ["graphics", "silhouettes", "true"] or parts == ["graphics", "silhouettes", "false"]:
        return True
    if _allowed_title(stripped, parts):
        return True
    if parts[0] == "turn" and len(parts) == 4:
        return parts[1] in {"x", "y", "z"} and FLOAT_RE.fullmatch(parts[2]) and INT_RE.fullmatch(parts[3])
    if parts[0] == "wait" and len(parts) == 2:
        return INT_RE.fullmatch(parts[1]) is not None
    if parts[0] == "matchmaker" and len(parts) == 4 and parts[2] == "to":
        return _safe_selection(parts[1]) and _safe_selection(parts[3])
    return False


def _send(*, port: int, command: str, timeout_seconds: float) -> dict[str, object]:
    query = urllib.parse.urlencode({"command": command})
    url = f"http://127.0.0.1:{port}/run?{query}"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
        payload = response.read().decode("utf-8")
    parsed = json.loads(payload)
    if parsed.get("error") is not None:
        raise RuntimeError(f"ChimeraX returned error for {command!r}: {parsed['error']}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Send one allowlisted command to a local ChimeraX REST endpoint.")
    parser.add_argument("--port", type=int)
    parser.add_argument("--session-manifest", type=Path)
    parser.add_argument("--command", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()

    try:
        port = _resolve_port(port=args.port, session_manifest=args.session_manifest)
    except ValueError as exc:
        parser.error(str(exc))
    if port < 1024 or port > 65535:
        parser.error("--port must be from 1024 to 65535")
    if not _allowed(args.command):
        raise SystemExit(f"Refusing non-allowlisted ChimeraX command: {args.command!r}")
    command_log_path = _session_command_log_path(args.session_manifest)
    try:
        response = _send(port=port, command=args.command, timeout_seconds=args.timeout_seconds)
    except Exception as exc:
        _append_command_log(
            command_log_path=command_log_path,
            port=port,
            command=args.command,
            status="failed",
            error=str(exc),
        )
        raise
    _append_command_log(
        command_log_path=command_log_path,
        port=port,
        command=args.command,
        status="accepted",
        response=response,
    )
    json.dump(response, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
