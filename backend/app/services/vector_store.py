import os
import pickle
import hashlib
import logging
from typing import List, Dict, Any, Set
import numpy as np
import faiss
from backend.app.config import settings

logger = logging.getLogger("documind.vector_store")


class VectorStore:
    def __init__(
        self,
        model_name: str = settings.embedding_model,
        index_path: str = settings.vector_store_path,
        similarity_threshold: float = settings.similarity_threshold
    ):
        self.model_name = model_name
        self.index_path = index_path
        self.similarity_threshold = similarity_threshold

        self._model = None
        self.dimension = None
        self.index = None
        self.metadata_store: List+
        [Dict[str, Any]] = []
        # Set of SHA-256 hashes of full raw file bytes — prevents duplicate uploads
        self.indexed_file_hashes: Set[str] = set()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def _ensure_index(self):
        if self.index is None:
            if self.dimension is None:
                self.dimension = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(self.dimension)

    # ------------------------------------------------------------------
    # File-level duplicate guard
    # ------------------------------------------------------------------
    @staticmethod
    def compute_file_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def is_duplicate_file(self, data: bytes) -> bool:
        return self.compute_file_hash(data) in self.indexed_file_hashes

    def register_file_hash(self, data: bytes) -> str:
        h = self.compute_file_hash(data)
        self.indexed_file_hashes.add(h)
        return h

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embeds chunks and adds them to the FAISS index with metadata."""
        if not chunks:
            return 0

        self._ensure_index()
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.index.add(embeddings)
        self.metadata_store.extend(chunks)
        logger.info("Added %d chunks (index total: %d)", len(chunks), self.index.ntotal)
        return len(chunks)

    # ------------------------------------------------------------------
    # Search with over-fetch + text deduplication
    # ------------------------------------------------------------------
    @staticmethod
    def _text_key(text: str) -> str:
        return hashlib.md5(text.strip().lower().encode()).hexdigest()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top_k *unique* chunks (by text hash) most relevant to query.
        Over-fetches top_k * 4 candidates to handle duplicates from repeated uploads.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

        # Over-fetch to have room to deduplicate
        candidates = min(top_k * 4, self.index.ntotal)
        scores, indices = self.index.search(query_vector, candidates)

        seen_text_keys: Set[str] = set()
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < self.similarity_threshold:
                continue

            item = dict(self.metadata_store[idx])
            key = self._text_key(item["text"])

            if key in seen_text_keys:
                continue  # skip duplicate passage

            seen_text_keys.add(key)
            item["similarity_score"] = float(score)
            results.append(item)

            if len(results) == top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str = None):
        """Persists the FAISS index, metadata store, and file hashes to disk."""
        save_path = path or self.index_path
        os.makedirs(save_path, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, os.path.join(save_path, "index.faiss"))

        with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "metadata": self.metadata_store,
                "file_hashes": self.indexed_file_hashes
            }, f)

    def load(self, path: str = None) -> bool:
        """Loads the FAISS index, metadata store, and file hashes from disk."""
        load_path = path or self.index_path
        index_file = os.path.join(load_path, "index.faiss")
        meta_file = os.path.join(load_path, "metadata.pkl")

        if not (os.path.exists(index_file) and os.path.exists(meta_file)):
            return False

        self.index = faiss.read_index(index_file)
        with open(meta_file, "rb") as f:
            data = pickle.load(f)
            # Support both old list format and new dict format
            if isinstance(data, dict):
                self.metadata_store = data.get("metadata", [])
                self.indexed_file_hashes = data.get("file_hashes", set())
            else:
                self.metadata_store = data
                self.indexed_file_hashes = set()

        if self.index.ntotal > 0:
            self.dimension = self.index.d

        logger.info(
            "Loaded index: %d chunks, %d known files",
            self.index.ntotal, len(self.indexed_file_hashes)
        )
        return True

    def clear(self):
        """Resets the in-memory index, metadata store, and file hash registry."""
        self.index = None
        self.metadata_store = []
        self.indexed_file_hashes = set()
