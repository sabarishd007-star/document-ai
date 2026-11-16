import pytest
from backend.app.services.chunking import chunk_text
from backend.app.utils.errors import DocuMindException

def test_chunk_text_basic():
    text = "Hello world! " * 50
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all("chunk_id" in c and "text" in c for c in chunks)
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1

def test_chunk_text_short():
    text = "Short sentence."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text

def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []

def test_chunk_text_invalid_params():
    with pytest.raises(DocuMindException) as exc1:
        chunk_text("test", chunk_size=0)
    assert "positive integer" in str(exc1.value.message)

    with pytest.raises(DocuMindException) as exc2:
        chunk_text("test", chunk_size=100, chunk_overlap=100)
    assert "strictly less than" in str(exc2.value.message)

    with pytest.raises(DocuMindException) as exc3:
        chunk_text("test", chunk_size=100, chunk_overlap=-1)
    assert "non-negative" in str(exc3.value.message)

def test_chunk_text_metadata_preservation():
    metadata = {"doc_id": "file_123", "filename": "sample.pdf"}
    chunks = chunk_text("Doc content here", chunk_size=100, chunk_overlap=20, metadata=metadata)
    assert len(chunks) == 1
    assert chunks[0]["doc_id"] == "file_123"
    assert chunks[0]["filename"] == "sample.pdf"
    assert chunks[0]["chunk_id"] == "file_123_chunk_0"

