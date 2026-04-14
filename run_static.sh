#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QLEARN="${PROJECT_ROOT}/environment_static/Q_learn.py"
PYTHON=/Users/gtheming/miniconda3/envs/cs4100/bin/python3

mkdir -p "${PROJECT_ROOT}/results/default"

PYTHONPATH="${PROJECT_ROOT}" "${PYTHON}" -u "${QLEARN}" "$@" --outdir "${PROJECT_ROOT}/results/default"