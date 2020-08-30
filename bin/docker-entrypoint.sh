#!/usr/bin/env bash

set -o errexit  # Exit on error. Append || true if you expect an error.
set -o noclobber  # Don't allow overwriting files.
set -o nounset  # Don't allow use of undefined vars. Use ${VAR:-} to use an undefined VAR.

action="${1}"; shift

case "${action}" in

  # run [docker-compose run --service-ports ds exec bash] to enter the container machine
  exec)
    exec "$@"
  ;;

  jupyter)
    exec ./bin/start-jupyter.sh
  ;;

  update_requirements)
    bin/update-requirements.sh
  ;;

  *)
    echo "Invalid action: ${action}"
    exit 1
  ;;

esac
