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

if ! bin="$(find_chimerax)"; then
  printf 'FAIL: ChimeraX executable not found. Set CHIMERAX_BIN or install ChimeraX.\n' >&2
  exit 1
fi

printf 'PASS: ChimeraX executable: %s\n' "$bin"
if "$bin" --version >/tmp/chimerax-structure-review-version.txt 2>&1; then
  sed 's/^/INFO: /' /tmp/chimerax-structure-review-version.txt
else
  printf 'WARN: ChimeraX --version did not return cleanly; executable still exists.\n'
fi
