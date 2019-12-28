import numpy as np
from elastiknn.elastiknn_pb2 import SIMILARITY_JACCARD, SparseBoolVector
from elastiknn.models import ElastiKnnModel

from ann_benchmarks.algorithms.base import BaseANN


class ElastiKnnExact(BaseANN):

    def __init__(self, metric, n_shards=1, start_es: bool = False):
        self._model = ElastiKnnModel(algorithm='exact', metric=metric)
        self.name = 'elastiknn-exact'
        self.n_shards = n_shards
        self.batch_res = None
        self._dim = None

        if start_es:
            from subprocess import run
            if run("curl localhost:9200", shell=True).returncode != 0:
                run("service elasticsearch start", shell=True, check=True)

    def fit(self, X):
        # ann-benchmarks represents a sparse matrix as a list of lists of indices.
        if self._model._sim == SIMILARITY_JACCARD:
            self._dim = max(map(max, X)) + 1
            X = [SparseBoolVector(total_indices=self._dim, true_indices=x) for x in X]
        self._model.fit(X, recreate_index=True, shards=self.n_shards)

    def query(self, q, n):
        if self._model._sim == SIMILARITY_JACCARD:
            q = SparseBoolVector(total_indices=self._dim, true_indices=q)
        return self._model.kneighbors([q], n_neighbors=n, return_distance=False)[0]

    def batch_query(self, X, n):
        # self.batch_res = self._model.kneighbors(X, n_neighbors=n, return_distance=False)
        pass

    def get_batch_results(self):
        return self.batch_res
