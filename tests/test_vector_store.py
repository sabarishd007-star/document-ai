import os
import shutil
import pytest
import numpy as np
from unittest.mock import MagicMock
from backend.app.services.vector_store import VectorStore

@pytest.fixture
def mock_vector_store(tmp_path):
    store = VectorStore(index_path=str(tmp_path / "faiss_test"))
    
    # Mock sentence transformer model to run instantaneously
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 4
    
    def mock_encode(texts, convert_to_numpy=True, normalize_embeddings=True):
        vectors = []
        for text in texts:
            # Deterministic mock vectors
            if "python" in text.lower():
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "javascript" in text.lower():
                vectors.append([0.0, 1.0, 0.0, 0.0])
            else:
                vectors.append([0.25, 0.25, 0.25, 0.25])
        return np.array(vectors, dtype=np.float32)

    mock_model.encode = mock_encode
    store._model = mock_model
    store.dimension = 4
    return store

def test_add_and_search_chunks(mock_vector_store):
    chunks = [
        {"chunk_id": "c1", "text": "Python is a backend programming language."},
        {"chunk_id": "c2", "text": "JavaScript is used widely for web development."}
    ]
    
    added_count = mock_vector_store.add_chunks(chunks)
    assert added_count == 2
    assert mock_vector_store.index.ntotal == 2

    # Query matching Python
    results = mock_vector_store.search("python programming", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"
    assert "similarity_score" in results[0]

def test_save_and_load_persistence(mock_vector_store, tmp_path):
    chunks = [
        {"chunk_id": "c1", "text": "Python language data."}
    ]
    mock_vector_store.add_chunks(chunks)
    mock_vector_store.save()

    # Create new store and load
    new_store = VectorStore(index_path=str(tmp_path / "faiss_test"))
    new_store._model = mock_vector_store.model
    loaded = new_store.load()

    assert loaded is True
    assert new_store.index.ntotal == 1
    assert len(new_store.metadata_store) == 1
    assert new_store.metadata_store[0]["chunk_id"] == "c1"

def test_search_empty_store(mock_vector_store):
    results = mock_vector_store.search("anything")
    assert results == []

