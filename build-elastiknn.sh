#!/bin/bash
set -e

docker build --no-cache -t ann-benchmarks-elastiknn -f install/Dockerfile.elastiknn .
docker run -v $(pwd)/ann_benchmarks:/home/app/ann_benchmarks --rm elastiknn
