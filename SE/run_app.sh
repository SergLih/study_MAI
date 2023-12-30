#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

PROCESSOR_COUNT=$(nproc)
GUNICORN_WORKER_COUNT=$(( PROCESSOR_COUNT * 2 + 1 ))

function main()
{
  if ! gunicorn --workers ${GUNICORN_WORKER_COUNT} -b 0.0.0.0:9808 gun_app:application ; then
    echo "Failed to run gunicorn..."
    return 1
  fi
}

main $@
