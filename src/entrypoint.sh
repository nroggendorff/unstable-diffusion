#!/bin/bash
set -e
python3 -m src."$1" "${@:2}"
