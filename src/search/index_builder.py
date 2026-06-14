import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import EMBEDDING_BATCH_SIZE, EMBEDDINGS_PATH, FAISS_INDEX_PATH, INDEX_MAP_PATH
from src.data.schema import CarAd
from src.search.embedder import Embedder

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


class SimpleIndex:
    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def search(self, query_vectors: np.ndarray, k: int):
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        scores = query_vectors @ self.embeddings.T
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        if self.embeddings.size == 0:
            empty_scores = np.full((query_vectors.shape[0], k), -1.0, dtype=np.float32)
            empty_pos = np.full((query_vectors.shape[0], k), -1, dtype=np.int64)
            return empty_scores, empty_pos

        k = min(k, self.embeddings.shape[0])
        top_positions = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top_positions, axis=1)

        return top_scores, top_positions


def build_faiss_index(embeddings: np.ndarray):
    if faiss is None:
        return SimpleIndex(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def build_id_map(ads: List[CarAd]) -> List[str]:
    return [str(ad.ad_id) for ad in ads]


def save_index(
    index,
    id_map: List[str],
    embeddings: np.ndarray,
    index_path: Path = FAISS_INDEX_PATH,
    map_path: Path = INDEX_MAP_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if faiss is not None:
        faiss.write_index(index, str(index_path))
    map_path.write_text(json.dumps(id_map, ensure_ascii=False), encoding="utf-8")
    np.save(str(embeddings_path), embeddings)


def load_index(
    index_path: Path = FAISS_INDEX_PATH,
    map_path: Path = INDEX_MAP_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> Tuple[object, List[str]]:
    id_map = json.loads(map_path.read_text(encoding="utf-8"))
    if faiss is not None and index_path.exists():
        index = faiss.read_index(str(index_path))
        return index, id_map
    if embeddings_path.exists():
        embeddings = np.load(str(embeddings_path))
        return SimpleIndex(embeddings), id_map
    raise FileNotFoundError("No searchable index artifacts found")


def build_and_save(
    ads: List[CarAd],
    embedder: Embedder,
    index_path: Path = FAISS_INDEX_PATH,
    map_path: Path = INDEX_MAP_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> None:
    embeddings = embedder.encode_ads(ads, batch_size=batch_size)
    index = build_faiss_index(embeddings)
    id_map = build_id_map(ads)
    save_index(index, id_map, embeddings, index_path, map_path, embeddings_path)
