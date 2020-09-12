"""
ann-benchmarks interfaces for elastiknn: https://github.com/alexklibisz/elastiknn
Uses the elastiknn python client
To install a local copy of the client: pip install --upgrade -e /path/to/elastiknn/client-python/
"""

import numpy as np
from elastiknn.api import Vec
from elastiknn.models import ElastiknnModel
from elastiknn.utils import dealias_metric

from ann_benchmarks.algorithms.base import BaseANN

from subprocess import run


def start_es():
    if run("curl -s -I localhost:9200", shell=True).returncode != 0:
        print("Starting elasticsearch service...")
        run("service elasticsearch start", shell=True, check=True)


class Exact(BaseANN):

    def __init__(self, metric: str, dimension: int):
        self.name = f"eknn-exact-metric={metric}_dimension={dimension}"
        self.metric = metric
        self.dimension = dimension
        self.model = ElastiknnModel("exact", dealias_metric(metric))
        start_es()

    def _handle_sparse(self, X):
        # convert list of lists of indices to sparse vectors.
        return [Vec.SparseBool(x, self.dimension) for x in X]

    def fit(self, X):
        if self.metric in {'jaccard', 'hamming'}:
            return self.model.fit(self._handle_sparse(X), shards=1)[0]
        else:
            return self.model.fit(X, shards=1)

    def query(self, q, n):
        if self.metric in {'jaccard', 'hamming'}:
            return self.model.kneighbors(self._handle_sparse([q]), n)[0]
        else:
            return self.model.kneighbors(np.expand_dims(q, 0), n)[0]


class L2Lsh(BaseANN):

    def __init__(self, L: int, k: int, w: int):
        self.name_prefix = f"eknn-l2lsh-L={L}-k={k}-w={w}"
        self.name = None   # set based on query args.
        self.model = ElastiknnModel("lsh", "l2", mapping_params=dict(L=L, k=k, w=w))
        self.X_max = 1.0
        self.query_params = dict()
        start_es()

    def fit(self, X):
        self.X_max = X.max()
        return self.model.fit(X / self.X_max, shards=1)

    def set_query_arguments(self, candidates: int, probes: int):
        self.name = f"{self.name_prefix}_candidates={candidates}_probes={probes}"
        self.query_params = dict(candidates=candidates, probes=probes)

    def query(self, q, n):
        return self.model.kneighbors(np.expand_dims(q, 0) / self.X_max, n, query_params=self.query_params)[0]



