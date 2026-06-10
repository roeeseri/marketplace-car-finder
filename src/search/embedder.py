from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from src.data.schema import CarAd


def build_ad_text(ad: CarAd) -> str:
    parts = [
        ad.make,
        ad.model,
        str(ad.year) if ad.year else None,
        ad.gear_type,
        ad.fuel_type,
        ad.location,
        ad.description,
    ]
    return " ".join(p.strip() for p in parts if p and str(p).strip())


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vectors, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])

    def encode_ads(
        self, ads: List[CarAd], batch_size: int = EMBEDDING_BATCH_SIZE
    ) -> np.ndarray:
        texts = [build_ad_text(ad) for ad in ads]
        return self.encode(texts, batch_size=batch_size)
