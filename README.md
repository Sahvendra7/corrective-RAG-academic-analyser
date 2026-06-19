# 📚 Corrective RAG (CRAG) Academic Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Enabled-orange.svg)](https://python.langchain.com/docs/langgraph)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-Pro-blueviolet.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Corrective Retrieval Augmented Generation (CRAG) Academic Paper Analyzer built with **LangGraph**, **FAISS**, **Google Gemini**, and **FastAPI**.

This project implements the [CRAG framework (Yan et al., 2024)](https://arxiv.org/abs/2401.15884), moving beyond naive RAG by incorporating a retrieval evaluator to assess document relevance. If retrieved documents lack sufficient context, the agent intelligently triggers a web search (via Tavily) to fetch the missing information, ensuring high-fidelity, hallucination-free generation.


## ✨ Features

- **End-to-End Document Ingestion:** Automated downloading of arXiv papers, robust PyMuPDF parsing, and optimized text chunking.
- **Local Vector Database:** High-performance FAISS indexing coupled with `sentence-transformers` for fast semantic retrieval.
- **Agentic CRAG Pipeline (LangGraph):**
  - 📥 **Retriever:** Fetches top-k relevant chunks.
  - ⚖️ **Grader:** Strict evaluation of chunk relevance against the user query.
  - 🌐 **Web Search Fallback:** Tavily API integration for context enrichment when local retrieval fails.
  - 🧠 **Generator:** Powered by Google Gemini (via `langchain-google-genai`) for nuanced and accurate answer generation.
  - 🛡️ **Hallucination Checker:** Enforces grounding by verifying the generated answer against the retrieved context.
  - 🔄 **Query Rewriter:** Iteratively refines the query for better retrieval if hallucination is detected.
- **Production-Ready API:** FastAPI backend featuring Server-Sent Events (SSE) for real-time streaming and built-in rate limiting.
- **Interactive UI:** A sleek Streamlit frontend allowing users to chat with the agent and visualize the reasoning graph step-by-step.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Google Gemini API Key
- Tavily API Key

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sahvendraz/Corrective-RAG-Academic-Analyser.git
cd Corrective-RAG-Academic-Analyser

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install project dependencies
pip install -e .
```

### Configuration

Copy the example environment variables file and insert your API keys:

```bash
cp .env.example .env
```
Update `.env` with:
```env
GEMINI_API_KEY="your_google_gemini_api_key"
TAVILY_API_KEY="your_tavily_api_key"
```

## 🛠️ Usage

### 1. Data Ingestion (Optional)
To index new academic papers into your local FAISS vector store, run the ingestion pipeline:

```bash
python src/ingestion/arxiv_downloader.py
python src/ingestion/pdf_parser.py
python src/ingestion/chunker.py
python src/vectorstore/embeddings.py
```

### 2. Start the Backend API
Launch the FastAPI server (runs locally at `http://localhost:8000`):

```bash
uvicorn src.api.server:app --reload
```
*Tip: You can access the Swagger UI documentation at `http://localhost:8000/docs`.*

### 3. Launch the Streamlit UI
In a new terminal window, start the interactive chat interface:

```bash
streamlit run ui/app.py
```

## 🧪 Testing & Evaluation

Run the unit and integration tests using `pytest`:
```bash
pytest tests/
```

We leverage tools like `deepeval` and `ragas` for pipeline benchmarking. You can find evaluation scripts inside the `src/evaluation/` directory.

## 🏗️ Project Architecture

```text
├── configs/            # Application configuration files
├── data/               # Downloaded PDFs, parsed text, and vector stores
├── src/
│   ├── api/            # FastAPI server, routers, and SSE streaming
│   ├── evaluation/     # RAG benchmarking and testing scripts
│   ├── ingestion/      # Downloaders, parsers, and chunkers
│   ├── pipeline/       # LangGraph state definitions and node logic
│   ├── utils/          # Shared helpers and metadata managers
│   └── vectorstore/    # FAISS wrappers and embedding models
├── tests/              # Pytest integration and unit tests
├── ui/                 # Streamlit application frontend
└── notebooks/          # Jupyter notebooks for experimentation
```


## 📄 License

This project is licensed under the MIT License.
