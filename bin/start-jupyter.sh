#!/usr/bin/env bash

basePath=$(git rev-parse --show-toplevel)

# shellcheck disable=SC1090
source "${basePath}/bin/common.sh"

if test -f .env ; then
  # shellcheck disable=SC1091
  source .env
fi

jupyter notebook --allow-root
