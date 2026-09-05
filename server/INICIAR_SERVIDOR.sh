#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
chmod +x ./TridiCollectorServer 2>/dev/null || true
exec ./TridiCollectorServer --data "$HERE/DADOS_TVBOX"
