#!/bin/bash

set -eu

pip install .[all] "jax[cpu]<=0.4.10"

mypy -p tsax

coverage run --source tsax -m xmlrunner discover test || true

mkdir -p /coverage
cp -v .coverage.* /coverage/

mkdir -p /unittest
cp -v *.xml /unittest/
