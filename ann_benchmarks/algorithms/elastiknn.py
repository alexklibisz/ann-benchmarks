from logging import Logger

import numpy as np
from elastiknn.api import Vec
from elastiknn.models import ElastiknnModel
from elastiknn.utils import dealias_metric

from ann_benchmarks.algorithms.base import BaseANN

from subprocess import run


class ElastiknnWrapper(BaseANN):

    # Some development notes:
    #  To install a local copy of the elastiknn client: pip install --upgrade -e ../elastiknn/client-python/

    def __init__(self, metric: str, dimension: int, algorithm: str, mapping_params: dict = {}, query_params: dict = {}):
        self.name = f"eknn-{algorithm}-{metric}-{ElastiknnWrapper._dict_to_str(mapping_params)}-{ElastiknnWrapper._dict_to_str(query_params)}"
        self._dim = dimension
        self._metric = metric
        print(f"metric [{metric}], dimension [{dimension}], algorithm [{algorithm}], mapping_params [{mapping_params}], query_params [{query_params}]")

        # Check if Elasticsearch is running. Start it if not, assuming running in image built from Dockerfile.elastiknn.
        if run("curl -s -I localhost:9200", shell=True).returncode != 0:
            print("Starting elasticsearch service...")
            run("service elasticsearch start", shell=True, check=True)

        # algos.yaml is not specific to datasets, which can have different dimensions, so `k` is expressed as a
        # proportion of the dimension (`p`) and then converted to an int (`k`).
        if algorithm == 'permutation_lsh' and 'p' in mapping_params:
            mapping_params['k'] = int(dimension * mapping_params['p'])
            del mapping_params['p']

        self._model = ElastiknnModel(algorithm=algorithm, metric=metric, mapping_params=mapping_params, query_params=query_params)
        self._batch_res = None          # Defined in batch_query().
        self._transform = lambda x: x   # Defined in fit().

    def _dict_to_str(d: dict) -> str:
        return '-'.join([f"{k}_{v}" for k, v in d.items()])

    def _fix_sparse(self, X):
        # ann-benchmarks represents a sparse matrix as a list of lists of indices.
        return [Vec.SparseBool(x, self._dim) for x in X]

    def fit(self, X):
        if self._metric in {'jaccard', 'hamming'}:
            self._transform = self._fix_sparse()
        elif self._metric in {'euclidean'}:
            self._transform = lambda X_: X_ / X.max()
        return self._model.fit(self._transform(X), shards=1)

    def query(self, q, n):
        if self._metric in {'jaccard', 'hamming'}:
            X = self._fix_sparse([q])
        else:
            X = np.expand_dims(q, 0)
        return self._model.kneighbors(self._transform(X), n_neighbors=n, return_similarity=False)[0]

    def batch_query(self, X, n):
        if self._metric in {'jaccard', 'hamming'}:
            X = self._fix_sparse(X)
        self._batch_res = self._model.kneighbors(X, n_neighbors=n, return_similarity=False)

    def get_batch_results(self):
        return self._batch_res
