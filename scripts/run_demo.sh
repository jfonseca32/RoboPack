#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$root/build/robopack" ]]; then
    printf '%s\n' 'Build first: bash scripts/build.sh' >&2
    exit 1
fi
exec "$root/build/robopack" --demo --config "$root/config/robot.conf" "$@"
