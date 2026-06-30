#!/usr/bin/env bash
# Write .env from the current shell for every key listed in .env.example.
#
# For each KEY=... line in .env.example, runs `printenv KEY` and writes KEY=<value>
# to .env. Comment lines and blank lines from .env.example are copied through.
# If a key is not set in the environment, the placeholder value from .env.example
# is kept and a warning is printed to stderr.
#
# Usage (from anywhere):
#   ./scripts/write_env_from_shell.sh          # writes .env (prompts if it exists)
#   ./scripts/write_env_from_shell.sh --force  # overwrite without prompting
#
# Tip: export variables in your shell first, or source an existing .env:
#   set -a && source .env && set +a
#   ./scripts/write_env_from_shell.sh

set -euo pipefail

force=false
if [[ "${1:-}" == "--force" ]]; then
  force=true
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--force]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="${ROOT}/.env.example"
OUTPUT="${ROOT}/.env"

if [[ ! -f "${EXAMPLE}" ]]; then
  echo "error: ${EXAMPLE} not found" >&2
  exit 1
fi

if [[ -f "${OUTPUT}" && "${force}" != true ]]; then
  echo "error: ${OUTPUT} already exists (use --force to overwrite)" >&2
  exit 1
fi

quote_if_needed() {
  local value="$1"
  if [[ "${value}" == *[$' \t#"$\'\\']* ]]; then
    printf '%q' "${value}"
  else
    printf '%s' "${value}"
  fi
}

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

while IFS= read -r line || [[ -n "${line}" ]]; do
  # Copy comments and blank lines verbatim.
  if [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]]; then
    printf '%s\n' "${line}" >>"${tmp}"
    continue
  fi

  if [[ ! "${line}" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
    printf '%s\n' "${line}" >>"${tmp}"
    continue
  fi

  key="${BASH_REMATCH[1]}"
  rest="${BASH_REMATCH[2]}"

  # Drop inline comments from the example value (outside quotes).
  example_value="${rest%%#*}"
  example_value="${example_value%"${example_value##*[![:space:]]}"}"

  if [[ "${example_value}" =~ ^\"(.*)\"$ ]]; then
    example_value="${BASH_REMATCH[1]}"
  elif [[ "${example_value}" =~ ^\'(.*)\'$ ]]; then
    example_value="${BASH_REMATCH[1]}"
  fi

  if value="$(printenv "${key}" 2>/dev/null)"; then
    printf '%s=%s\n' "${key}" "$(quote_if_needed "${value}")" >>"${tmp}"
  else
    echo "warning: ${key} is not set in the environment; keeping .env.example placeholder" >&2
    printf '%s=%s\n' "${key}" "$(quote_if_needed "${example_value}")" >>"${tmp}"
  fi
done <"${EXAMPLE}"

mv "${tmp}" "${OUTPUT}"
trap - EXIT

echo "Wrote ${OUTPUT}"
