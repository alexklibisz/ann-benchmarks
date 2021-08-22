import logging
from urllib.error import URLError
from requests import Session
from time import sleep

from ann_benchmarks.algorithms.base import BaseANN


class LuceneHnsw(BaseANN):

    def __init__(self, metric, dimension, params):
        self.search_strategy = {"angular": "dotproducthnsw", "euclidean": "euclideanhnsw"}[metric]
        self.dimension = dimension
        self.max_connections = params["M"]
        self.beam_width = params["efConstruction"]
        self.name = f"lucenehnsw-{self.search_strategy}-{self.dimension}-{self.max_connections}-{self.beam_width}"
        self.index = str(abs(hash(self.name)))
        self.url = "http://localhost:8080"
        self.session = Session()

        self.num_seed = None  # set in set_query_arguments
        self.batch_res = None # set in batch_query

        # Make sure the server is running.
        for i in range(30):
            try:
                self.session.get(f"{self.url}/ready").raise_for_status()
                break
            except:
                sleep(1)
        else:
            raise RuntimeError("Failed to connect to local server")
        

    def fit(self, X):
        index_params = {
            "dims": self.dimension,
            "searchStrategy": self.search_strategy,
            "maxConnections": self.max_connections,
            "beamWidth": self.beam_width
        }
        res = self.session.put(f"{self.url}/{self.index}", json=index_params)
        assert res.status_code == 200, ("Failed to create index", res)
        
        batch_size = 1000
        for i in range(0, len(X), batch_size):
            X_ = [[round(float(f), 8) for f in a] for a in X[i:i+batch_size]]
            res = self.session.post(f"{self.url}/{self.index}", json=X_)
            assert res.status_code == 201, ("Failed to index vectors", res)

        res = self.session.post(f"{self.url}/{self.index}/close", timeout=100 * 60)
        assert res.status_code == 200, ("Failed to close index", res)
        println("Finished indexing")

    def set_query_arguments(self, ef):
        self.num_seed = ef

    def query(self, q, n):
        search_body = {
            "k": n,
            "params": {"numSeed": self.num_seed},
            "vector": [round(float(f), 8) for f in q]
        }
        res = self.session.post(f"{self.url}/{self.index}/search", json=search_body)
        assert res.status_code == 200, ("Failed to run query", res)
        return res.json()

    def batch_query(self, X, n):
        self.batch_res = [self.query(arr, n) for arr in X]

    def get_batch_results(self):
        return sel.batch_res