#!/bin/bash
cd "$(dirname "$0")"
python -m enviroment_static.Q_learn "$@"
