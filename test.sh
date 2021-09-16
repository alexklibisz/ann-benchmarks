#!/bin/bash
set -e
sudo chown -R $(whoami):$(whoami) results/
cd ../elastiknn && ./gradlew shadowJar && cd -
cp ../elastiknn/elastiknn-ann-benchmarks/build/libs/ann-benchmarks-7.14.0.0-all.jar .
docker build --build-arg cachebust=$RANDOM -t ann-benchmarks-elastiknn -f install/Dockerfile.elastiknn .
python run.py --algorithm elastiknn-l2lsh --dataset fashion-mnist-784-euclidean --runs 1 --count 100 --force --max-n-algorithm 1
sudo chown -R $(whoami):$(whoami) results/
python plot.py --dataset fashion-mnist-784-euclidean --recompute --count 100 -o out.png