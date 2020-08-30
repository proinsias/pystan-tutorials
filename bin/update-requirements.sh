#!/usr/bin/env bash

basePath=$(git rev-parse --show-toplevel)

# shellcheck disable=SC1090
source "${basePath}/bin/common.sh"

CUSTOM_COMPILE_COMMAND="docker-compose run pystan update_requirements"
export CUSTOM_COMPILE_COMMAND

pip-compile --annotate --header --upgrade --verbose \
    --output-file requirements.txt requirements.in

safety check --file requirements.txt --full-report
