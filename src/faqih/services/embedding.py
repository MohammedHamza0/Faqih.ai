"""Embedding model service for Arabic text."""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Wrapper around SentenceTransformers for Arabic text embedding.

    Supports batch encoding with caching.
    """

    def __init__(self, model_name: str = "aubmindlab/bert-base-arabertv2", device: str | None = None):
        self._model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None
        self._dim: int = 768

    def load(self):
        """Load the embedding model."""
        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        # Get actual dimension from a test encode
        test_emb = self._model.encode(["test"], show_progress_bar=False)
        self._dim = test_emb.shape[1]
        logger.info("Model loaded. Embedding dim: %d", self._dim)

    @property
    def model(self) -> SentenceTransformer:
        if not self._model:
            raise RuntimeError("Embedding model not loaded. Call load() first.")
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text(s) into dense vectors.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: L2 normalize vectors

        Returns:
            numpy array of shape (n_texts, dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )
        return embeddings

    def encode_single(self, text: str, normalize: bool = True) -> list[float]:
        """Encode a single text and return as list of floats."""
        embedding = self.encode(text, normalize=normalize)
        return embedding[0].tolist()

    @staticmethod
    def text_hash(text: str) -> str:
        """Generate SHA256 hash for text (used as cache key)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
