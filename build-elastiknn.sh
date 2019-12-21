#!/bin/bash
set -e

ANN_DIR=$(pwd)
EKN_DIR=$HOME/Documents/dev/elastiknn

cd $EKN_DIR && ./gradlew clean assemble 
cp $EKN_DIR/es74x/build/distributions/elastiknn-0.0.0-SNAPSHOT_es7.4.0.zip $ANN_DIR/elastiknn.zip

docker build -t elastiknn -f install/Dockerfile.elastiknn .
docker run -v $(pwd)/ann_benchmarks:/home/app/ann_benchmarks --rm elastiknn
