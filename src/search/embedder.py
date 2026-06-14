from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

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
        self.model_name = model_name
        self._backend = self._load_backend()

    def _load_backend(self):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            return SentenceTransformer(self.model_name)
        except Exception:
            return None

    @staticmethod
    def _fallback_encode(texts: List[str], dim: int = 384) -> np.ndarray:
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[\w\u0590-\u05FF']+", (text or "").lower())
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest, "little") % dim
                vectors[row, idx] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors

    def encode(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        if self._backend is None:
            return self._fallback_encode(texts)

        vectors = self._backend.encode(
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
