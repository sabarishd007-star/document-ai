import pytest
from backend.app.main import root, health_check, get_index_status, query_documents, QueryRequest, vector_store

@pytest.fixture(autouse=True)
def clean_vector_store():
    vector_store.clear()

def test_root_and_health():
    res = root()
    assert res["status"] == "online"
    assert "DocuMind API" in res["app"]

    health = health_check()
    assert health["status"] == "healthy"

def test_api_status():
    status = get_index_status()
    assert status["total_chunks_indexed"] == 0
    assert status["similarity_threshold"] > 0

def test_query_no_documents():
    req = QueryRequest(query="What is DocuMind?")
    res = query_documents(req)
    assert "No relevant content found" in res["answer"]
    assert res["citations"] == []
