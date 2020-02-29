from elastiknn.elastiknn_pb2 import SIMILARITY_JACCARD, SparseBoolVector
from elastiknn.models import ElastiKnnModel

from ann_benchmarks.algorithms.base import BaseANN


class ElastiKnnExact(BaseANN):

    def __init__(self, metric, num_shards=1, use_cache: bool = True, start_es: bool = False):
        self._model = ElastiKnnModel(algorithm='exact', metric=metric)
        self.name = 'elastiknn-exact'
        self.num_shards = num_shards
        self.batch_res = None
        self.use_cache = use_cache
        self._dim = None

        print(f"Running {self.name} with metric {metric} and {num_shards} shards")

        if start_es:
            from subprocess import run
            if run("curl localhost:9200", shell=True).returncode != 0:
                print("Starting elasticsearch service...")
                run("service elasticsearch start", shell=True, check=True)

    def fit(self, X):
        # ann-benchmarks represents a sparse matrix as a list of lists of indices.
        if self._model._sim == SIMILARITY_JACCARD:
            self._dim = max(map(max, X)) + 1
            X = [SparseBoolVector(total_indices=self._dim, true_indices=x) for x in X]
        return self._model.fit(X, recreate_index=True, shards=self.num_shards)

    def query(self, q, n):
        if self._model._sim == SIMILARITY_JACCARD:
            q = SparseBoolVector(total_indices=self._dim, true_indices=q)
        return self._model.kneighbors([q], n_neighbors=n, return_distance=False, use_cache=self.use_cache)[0]

    def batch_query(self, X, n):
        self.batch_res = self._model.kneighbors(X, n_neighbors=n, return_distance=False, use_cache=self.use_cache)

    def get_batch_results(self):
        return self.batch_res


class ElastiKnnLsh(BaseANN):

    def __init__(self, metric, num_shards=1, num_bands: int = 10, num_rows: int = 2, use_cache: bool = True,
                 start_es: bool = False):
        self._model = ElastiKnnModel(algorithm='lsh', metric=metric, algorithm_params=dict(num_bands=num_bands, num_rows=num_rows))
        self.name = 'elastiknn-lsh'
        self.num_shards = num_shards
        self.batch_res = None
        self.use_cache = use_cache
        self._dim = None

        if start_es:
            from subprocess import run
            if run("curl -s localhost:9200", shell=True).returncode != 0:
                print("Starting elasticsearch service...")
                run("service elasticsearch start", shell=True, check=True)

    def fit(self, X):
        # ann-benchmarks represents a sparse matrix as a list of lists of indices.
        if self._model._sim == SIMILARITY_JACCARD:
            self._dim = max(map(max, X)) + 1
            X = [SparseBoolVector(total_indices=self._dim, true_indices=x) for x in X]
        return self._model.fit(X, recreate_index=True, shards=self.num_shards)

    def query(self, q, n):
        if self._model._sim == SIMILARITY_JACCARD:
            q = SparseBoolVector(total_indices=self._dim, true_indices=q)
        return self._model.kneighbors([q], n_neighbors=n, return_distance=False, use_cache=self.use_cache)[0]

    def batch_query(self, X, n):
        self.batch_res = self._model.kneighbors(X, n_neighbors=n, return_distance=False, use_cache=self.use_cache)

    def get_batch_results(self):
        return self.batch_res
