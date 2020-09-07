from logging import Logger

import numpy as np
from elastiknn.api import Vec
from elastiknn.models import ElastiknnModel

from ann_benchmarks.algorithms.base import BaseANN

from subprocess import run


class ElastiknnWrapper(BaseANN):

    # Some development notes:
    #  To install a local copy of the elastiknn client: pip install --upgrade -e ../elastiknn/client-python/

    def __init__(self, algorithm: str, metric: str, mapping_params: dict, query_params: dict):
        self._metric = metric
        self._logger = Logger("Elastiknn")
        self._logger.info(f"algorithm [{algorithm}], metric [{metric}], mapping_params [{mapping_params}], query_params [{query_params}]")

        # Attempt to start elasticsearch, assuming running in image built from Dockerfile.elastiknn.
        if run("curl localhost:9200", shell=True).returncode != 0:
            print("Starting elasticsearch service...")
            run("service elasticsearch start", shell=True, check=True)

        self._model = ElastiknnModel(algorithm=algorithm, metric=metric, mapping_params=mapping_params, query_params=query_params)

        self._dim = None        # Defined in fit().
        self._batch_res = None  # Defined in batch_query().

    @staticmethod
    def _fix_sparse(X):
        # ann-benchmarks represents a sparse matrix as a list of lists of indices.
        dim = max(map(max, X)) + 1
        return dim, [Vec.SparseBool(x, dim) for x in X]

    def fit(self, X):
        if self._metric in {'jaccard', 'hamming'}:
            self._dim, X = ElastiknnWrapper._fix_sparse(X)
        else:
            self._dim = X.shape[-1]
        return self._model.fit(X, shards=1)

    def query(self, q, n):
        if self._metric in {'jaccard', 'hamming'}:
            _, X = ElastiknnWrapper._fix_sparse([q])
        else:
            X = np.expand_dims(q, 0)
        return self._model.kneighbors(X, n_neighbors=n, return_similarity=False, allow_missing=True)[0]

    def batch_query(self, X, n):
        if self._metric in {'jaccard', 'hamming'}:
            _, X = ElastiknnWrapper._fix_sparse(X)
        self._batch_res = self._model.kneighbors(X, n_neighbors=n, return_similarity=False, allow_missing=True)

    def get_batch_results(self):
        return self._batch_res
