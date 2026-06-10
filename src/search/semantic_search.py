from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from config import FAISS_INDEX_PATH, INDEX_MAP_PATH, TOP_K
from src.search.embedder import Embedder
from src.search.index_builder import load_index


@dataclass
class SearchResult:
    ad_id: str
    score: float


class SemanticSearch:
    def __init__(self, index, id_map: List[str], embedder: Embedder) -> None:
        self._index = index
        self._id_map = id_map
        self._embedder = embedder

    def search(self, query: str, k: int = TOP_K) -> List[SearchResult]:
        query_vector = self._embedder.encode_query(query)
        scores, positions = self._index.search(query_vector, k)
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:
                continue
            results.append(SearchResult(ad_id=self._id_map[pos], score=float(score)))
        return results

    def search_batch(self, queries: List[str], k: int = TOP_K) -> List[List[SearchResult]]:
        query_vectors = self._embedder.encode(queries)
        scores_batch, positions_batch = self._index.search(query_vectors, k)
        output = []
        for scores, positions in zip(scores_batch, positions_batch):
            results = []
            for score, pos in zip(scores, positions):
                if pos == -1:
                    continue
                results.append(SearchResult(ad_id=self._id_map[pos], score=float(score)))
            output.append(results)
        return output


def load_semantic_search(
    index_path: Path = FAISS_INDEX_PATH,
    map_path: Path = INDEX_MAP_PATH,
    embedder: Embedder = None,
) -> SemanticSearch:
    if embedder is None:
        embedder = Embedder()
    index, id_map = load_index(index_path, map_path)
    return SemanticSearch(index, id_map, embedder)
