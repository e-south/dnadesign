#!/usr/bin/env bash

if command -v rg >/dev/null 2>&1; then
  return 0
fi

rg() {
  local fixed=0
  local grep_opts=()
  while (($#)); do
    case "$1" in
      -F)
        fixed=1
        shift
        ;;
      -q|-i|-n)
        grep_opts+=("$1")
        shift
        ;;
      -m)
        grep_opts+=("-m" "$2")
        shift 2
        ;;
      -m*)
        grep_opts+=("$1")
        shift
        ;;
      -qi|-iq)
        grep_opts+=("-q" "-i")
        shift
        ;;
      -Fq|-qF)
        fixed=1
        grep_opts+=("-q")
        shift
        ;;
      --)
        shift
        break
        ;;
      -*)
        grep_opts+=("$1")
        shift
        ;;
      *)
        break
        ;;
    esac
  done

  if (( fixed )); then
    command grep -F "${grep_opts[@]}" -- "$@"
  else
    command grep -E "${grep_opts[@]}" -- "$@"
  fi
}
