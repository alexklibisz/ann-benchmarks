"""
ann-benchmarks interfaces for elastiknn: https://github.com/alexklibisz/elastiknn
Uses the elastiknn python client
To install a local copy of the client: pip install --upgrade -e /path/to/elastiknn/client-python/
"""

import numpy as np
from elastiknn.models import ElastiknnModel

from ann_benchmarks.algorithms.base import BaseANN

from subprocess import run


# class Elastiknn(BaseANN):
#
#     def __init__(self, metric: str, dimension: int, algorithm: str, mapping_params: dict = {}, query_params: dict = {}):
#         self.name = f"eknn-{algorithm}-{metric}-{Elastiknn._dict_to_str(mapping_params)}-{Elastiknn._dict_to_str(query_params)}"
#         self._dim = dimension
#         self._metric = metric
#         print(f"metric [{metric}], dimension [{dimension}], algorithm [{algorithm}], mapping_params [{mapping_params}], query_params [{query_params}]")
#
#         # Check if Elasticsearch is running. Start it if not, assuming running in image built from Dockerfile.elastiknn.
#         if run("curl -s -I localhost:9200", shell=True).returncode != 0:
#             print("Starting elasticsearch service...")
#             run("service elasticsearch start", shell=True, check=True)
#
#         # algos.yaml is not specific to datasets, which can have different dimensions, so `k` is expressed as a
#         # proportion of the dimension (`p`) and then converted to an int (`k`).
#         if algorithm == 'permutation_lsh' and 'p' in mapping_params:
#             mapping_params['k'] = int(dimension * mapping_params['p'])
#             del mapping_params['p']
#
#         self._model = ElastiknnModel(algorithm=algorithm, metric=metric, mapping_params=mapping_params, query_params=query_params)
#         self._batch_res = None          # Defined in batch_query().
#         self._transform = lambda x: x   # Defined in fit().
#
#     def _dict_to_str(d: dict) -> str:
#         return '-'.join([f"{k}_{v}" for k, v in d.items()])
#
#     def _handle_sparse(self, X):
#         # ann-benchmarks represents a sparse matrix as a list of lists of indices.
#         return [Vec.SparseBool(x, self._dim) for x in X]
#
#     def fit(self, X):
#         if self._metric in {'jaccard', 'hamming'}:
#             self._transform = lambda X_: self._handle_sparse(X_)
#         elif self._metric in {'euclidean'}:
#             self._transform = lambda X_: X_ / X.max()
#         return self._model.fit(self._transform(X), shards=1)
#
#     def query(self, q, n):
#         if self._metric in {'jaccard', 'hamming'}:
#             X = self._handle_sparse([q])
#         else:
#             X = np.expand_dims(q, 0)
#         return self._model.kneighbors(self._transform(X), n_neighbors=n, return_similarity=False)[0]
#
#     def batch_query(self, X, n):
#         if self._metric in {'jaccard', 'hamming'}:
#             X = self._handle_sparse(X)
#         self._batch_res = self._model.kneighbors(X, n_neighbors=n, return_similarity=False)
#
#     def get_batch_results(self):
#         return self._batch_res


def start_es():
    if run("curl -s -I localhost:9200", shell=True).returncode != 0:
        print("Starting elasticsearch service...")
        run("service elasticsearch start", shell=True, check=True)


class L2Lsh(BaseANN):

    def __init__(self, L: int, k: int, w: int):
        self.name_prefix = f"eknn-l2lsh-L={L}-k={k}-w={w}"
        self.name = None   # set based on query args.
        self.model = ElastiknnModel("lsh", "l2", mapping_params=dict(L=L, k=k, w=w))
        self.X_max = 1.0
        start_es()

    def fit(self, X):
        self.X_max = X.max()
        return self.model.fit(X / self.X_max, shards=1)

    def set_query_arguments(self, candidates: int, probes: int):
        self.name = f"{self.name_prefix}_candidates={candidates}_probes={probes}"
        self.model._query_params = dict(candidates=candidates, probes=probes)

    def query(self, q, n):
        return self.model.kneighbors(np.expand_dims(q, 0) / self.X_max, n)[0]



