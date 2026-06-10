import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from config import EMBEDDING_BATCH_SIZE, EMBEDDINGS_PATH, FAISS_INDEX_PATH, INDEX_MAP_PATH
from src.data.schema import CarAd
from src.search.embedder import Embedder


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def build_id_map(ads: List[CarAd]) -> List[str]:
    return [str(ad.ad_id) for ad in ads]


def save_index(
    index: faiss.IndexFlatIP,
    id_map: List[str],
    embeddings: np.ndarray,
    index_path: Path = FAISS_INDEX_PATH,
    map_path: Path = INDEX_MAP_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    map_path.write_text(json.dumps(id_map, ensure_ascii=False), encoding="utf-8")
    np.save(str(embeddings_path), embeddings)


def load_index(
    index_path: Path = FAISS_INDEX_PATH,
    map_path: Path = INDEX_MAP_PATH,
) -> Tuple[faiss.IndexFlatIP, List[str]]:
    index = faiss.read_index(str(index_path))
    id_map = json.loads(map_path.read_text(encoding="utf-8"))
    return index, id_map


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
