#!/bin/bash
set -e

service elasticsearch start
python3 run_algorithm.py