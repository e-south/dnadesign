#!/usr/bin/env bash
set -euo pipefail

find_chimerax() {
  if [[ -n "${CHIMERAX_BIN:-}" && -x "${CHIMERAX_BIN:-}" ]]; then
    printf '%s\n' "$CHIMERAX_BIN"
    return 0
  fi
  if command -v ChimeraX >/dev/null 2>&1; then
    command -v ChimeraX
    return 0
  fi
  if command -v chimerax >/dev/null 2>&1; then
    command -v chimerax
    return 0
  fi
  local app
  for app in /Applications/ChimeraX*.app "$HOME"/Applications/ChimeraX*.app; do
    if [[ -x "$app/Contents/MacOS/ChimeraX" ]]; then
      printf '%s\n' "$app/Contents/MacOS/ChimeraX"
      return 0
    fi
  done
  return 1
}

version_hint() {
  local bin="$1"
  local app_root=""
  case "$bin" in
    */Contents/MacOS/ChimeraX)
      app_root="${bin%/Contents/MacOS/ChimeraX}"
      ;;
  esac
  if [[ -n "$app_root" && -f "$app_root/Contents/Info.plist" ]] && command -v plutil >/dev/null 2>&1; then
    plutil -extract CFBundleShortVersionString raw -o - "$app_root/Contents/Info.plist" 2>/dev/null || true
    return 0
  fi
  return 1
}

if ! bin="$(find_chimerax)"; then
  printf 'FAIL: ChimeraX executable not found. Set CHIMERAX_BIN or install ChimeraX.\n' >&2
  exit 1
fi

printf 'PASS: ChimeraX executable: %s\n' "$bin"
if hint="$(version_hint "$bin")" && [[ -n "$hint" ]]; then
  printf 'INFO: ChimeraX app version: %s\n' "$hint"
else
  printf 'INFO: ChimeraX version not checked; preflight does not launch the executable.\n'
fi
