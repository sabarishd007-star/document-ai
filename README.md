# 🧠 DocuMind

**DocuMind** is an intelligent, high-performance Document Analysis and Retrieval-Augmented Generation (RAG) assistant powered by **Google Gemini** and **FAISS**.

DocuMind lets you upload unstructured documents (`.pdf` and `.txt`), extracts and chunks their contents, indexes them into a high-speed vector database with dense sentence embeddings, and grounds Gemini's answers strictly on facts found in your documents with verified source citations.

---

## 🚀 Key Features

- **Document Ingestion**: Extracts text from `.pdf` (via `pypdf`) and `.txt` (UTF-8 & Latin-1) files up to 10MB.
- **Overlapping Semantic Chunking**: Configurable chunk size and overlap (default 500 chars / 100 overlap) to preserve sentence boundary context.
- **Dense Vector Search**: Powered by `sentence-transformers` (`all-MiniLM-L6-v2`) and `faiss-cpu` (cosine similarity with normalized inner product).
- **Grounded Gemini RAG**: Enforces strict hallucination-free generation using Google's official `google-genai` SDK (`gemini-2.5-flash`), returning verified citations with match percentages.
- **Interactive Web UI**: Modern, responsive dashboard served directly by FastAPI (`/ui`).
- **CLI Utility**: Command-line interface (`cli.py`) for uploading, querying, and managing your vector store without leaving the terminal.
- **Comprehensive Test Suite**: 17 unit and integration tests covering all services and API endpoints.

---

## 📁 Project Structure

```text
document-ai/
├── backend/
│   └── app/
│       ├── config.py              # Pydantic v2 application settings (.env)
│       ├── main.py                # FastAPI app with CORS, routes, & static UI mount
│       ├── services/
│       │   ├── ingestion.py       # File validation & text extraction
│       │   ├── chunking.py        # Sliding window text chunking with metadata
│       │   ├── vector_store.py    # FAISS index & sentence-transformer embeddings
│       │   └── rag.py             # Grounded prompt construction & Gemini generation
│       └── utils/
│           └── errors.py          # Custom DocuMindException & error handlers
├── frontend/                      # Web dashboard
│   ├── index.html                 # UI layout & chat view
│   ├── style.css                  # Modern dark theme styles
│   └── app.js                     # Frontend API interactions & citation badges
├── sample_docs/
│   └── sample_report.txt          # Ready-to-use sample document for testing
├── tests/                         # Pytest test suite
│   ├── test_api.py                # API endpoint unit tests
│   ├── test_chunking.py           # Chunking boundary & metadata tests
│   ├── test_ingestion.py          # Ingestion & decoding tests
│   └── test_vector_store.py       # FAISS indexing & search tests
├── cli.py                         # Command-line utility
├── requirements.txt               # Dependencies
├── .env.example                   # Environment configuration template
└── README.md
```

---

## 🛠️ Quickstart Setup

### 1. Configure Python Environment

Ensure you have Python 3.11+ installed:

```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Set Up Your Gemini API Key

Copy `.env.example` to `.env` and insert your [Google AI Studio API key](https://aistudio.google.com/):

```env
GEMINI_API_KEY=AIzaSy...your_actual_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.30
VECTOR_STORE_PATH=./data/faiss_index
```

---

## 🖥️ Running the Application

### Option A: Start the Web UI & API Server

```powershell
python cli.py run --reload
```

Open your browser:
- **Web UI:** [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Option B: Use the Command-Line Interface (CLI)

```powershell
# 1. Check current vector store status
python cli.py status

# 2. Upload and index a document
python cli.py upload sample_docs/sample_report.txt

# 3. Ask a question about the document
python cli.py query "What is the chunking strategy and embedding model used in DocuMind?"

# 4. Reset/clear the index
python cli.py reset
```

---

## 🧪 Running Tests

Execute the full automated test suite:

```powershell
python -m pytest -p no:asyncio -p no:anyio tests/
```

All 17 tests verify ingestion validation, text chunking, FAISS cosine similarity ranking, persistence, and FastAPI routes.

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status and links |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/api/status` | Vector store chunk count and configuration |
| `POST` | `/api/upload` | Multipart upload for `.pdf` and `.txt` files |
| `POST` | `/api/query` | Ask a question (`{"query": "...", "top_k": 4}`) |
| `DELETE` | `/api/index` | Reset/clear all indexed vectors and metadata |
| `GET` | `/ui` | Web dashboard interface |

