#!/bin/bash
set -e

docker build -t ann-benchmarks -f install/Dockerfile .
docker build -t ann-benchmarks-datasketch -f install/Dockerfile.datasketch . 
docker build --no-cache -t ann-benchmarks-elastiknn -f install/Dockerfile.elastiknn .
docker run -v $(pwd)/ann_benchmarks:/home/app/ann_benchmarks --rm ann-benchmarks-elastiknn
