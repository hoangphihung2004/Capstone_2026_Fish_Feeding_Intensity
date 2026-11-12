#!/usr/bin/env bash
# Manage the AquaFFIA web dashboard container.
# Usage: docker/run_web.sh [build | up | stop | restart | logs | status]
set -euo pipefail
cd "$(dirname "$0")"

PORT="${AQUAFFIA_PORT:-8001}"

case "${1:-up}" in
  build)
    docker compose build web
    ;;
  up)
    mkdir -p ../cache ../benchmarking/results ../samples/video ../samples/audio
    docker compose up -d web
    echo "Dashboard: http://192.168.55.1:${PORT}"
    ;;
  stop)
    docker compose stop web
    ;;
  restart)
    docker compose restart web
    ;;
  logs)
    docker compose logs -f --tail 100 web
    ;;
  status)
    docker compose ps -a
    ;;
  *)
    echo "Usage: $0 [build | up | stop | restart | logs | status]"
    exit 1
    ;;
esac
