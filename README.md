# Corrective RAG (CRAG) Academic Analyzer

Corrective Retrieval Augmented Generation (CRAG) Academic Paper Analyzer using LangGraph, FAISS, and FastAPI.

This project implements the [CRAG paper (Yan et al., 2024)](https://arxiv.org/abs/2401.15884) pipeline, which incorporates a retrieval evaluator to assess the quality of retrieved documents. If documents are irrelevant, the system triggers web searches using Tavily to find the missing context, ensuring highly accurate generation.

## Features

- **Document Ingestion:** Downloads academic papers from arXiv, parses PDFs using PyMuPDF, and creates text chunks.
- **Vector Store:** Local vector store using FAISS and `sentence-transformers` for fast semantic search.
- **Pipeline:** LangGraph-based state machine implementing CRAG:
  - **Retriever:** Fetches top-k chunks.
  - **Grader:** Evaluates retrieved chunks for relevance to the user's query.
  - **Web Search:** Fallback using Tavily API if chunks are graded irrelevant.
  - **Generator:** Uses Google Gemini (via `langchain-google-genai`) to generate an answer.
  - **Hallucination Checker:** Verifies that the answer is grounded in the retrieved documents.
  - **Rewriter:** Rewrites the query if hallucinated or poorly answered to retry retrieval.
- **API Backend:** Production-ready FastAPI with streaming SSE (Server-Sent Events) and rate limiting.
- **Frontend UI:** Interactive Streamlit interface to chat with the agent and view its reasoning graph in real time.

## Installation

```bash
# Clone the repository
git clone https://github.com/sahvendraz/Corrective-RAG-Academic-Analyser.git
cd Corrective-RAG-Academic-Analyser

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

## Setup

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Fill in your API keys in the `.env` file:
   - `GEMINI_API_KEY`: Get from Google AI Studio.
   - `TAVILY_API_KEY`: Get from Tavily.

## Usage

### 1. Ingestion (Optional: if you want to index new papers)
Download and chunk papers:
```bash
python src/ingestion/arxiv_downloader.py
python src/ingestion/pdf_parser.py
python src/ingestion/chunker.py
python src/vectorstore/embeddings.py
```

### 2. Run the API Server
Start the FastAPI server (runs on `http://localhost:8000`):
```bash
uvicorn src.api.server:app --reload
```

### 3. Run the Streamlit UI
In a separate terminal, start the UI:
```bash
streamlit run ui/app.py
```

## Testing
Run the integration tests using pytest:
```bash
pytest tests/
```

## Architecture

- `src/api/`: FastAPI server and routes
- `src/evaluation/`: Scripts to evaluate the pipeline against a ground-truth dataset
- `src/ingestion/`: Tools to download, parse, and chunk academic papers
- `src/pipeline/`: LangGraph definitions, states, and node logic
- `src/utils/`: Shared utilities (e.g., metadata handling)
- `src/vectorstore/`: FAISS wrappers and embedding scripts
- `ui/`: Streamlit interface
- `configs/`: Additional configuration files (if used)
