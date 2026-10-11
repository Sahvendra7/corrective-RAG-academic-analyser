# 📚 Corrective RAG (CRAG) Academic Analyzer

Question answering over a corpus of arXiv papers with **cited, page-level evidence**, built with
LangGraph, FAISS, sentence-transformers, Google Gemini and FastAPI, with a Streamlit UI.

It implements the corrective loop of [CRAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884):
retrieved passages are graded, the query is rewritten when retrieval is ambiguous, and a Tavily web
search is used when the papers do not contain the answer. Answers cite the passages they use with
`[n]` markers; the source list (paper, section, pages, link) is generated from passage metadata,
not written by the model, and a grounding check flags claims and numbers the sources do not support.
When no usable evidence is found the system says so instead of answering.

> A grounding check reduces, but does not eliminate, unsupported statements. Verify important
> claims against the cited passages.

## Contents
- [How it works](#how-it-works)
- [Getting started](#getting-started)
- [Building the index](#building-the-index)
- [Running the API and UI](#running-the-api-and-ui)
- [Configuration](#configuration)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

## How it works

```text
arXiv PDF ─► pdf_parser (reading order, headings, pages, de-hyphenation, header/footer removal)
          ─► sections   (canonical labels: abstract, method, results, references, appendix …)
          ─► chunker    (≤170 words, section-bounded, sentence overlap, page range; bibliography dropped)
          ─► FAISS      (all-MiniLM-L6-v2, cosine; manifest-validated data/index/)

question ─► retriever ─► grader (1 batched LLM call)
              relevant ────────────────────────────────► generator ─► grounding check ─► answer
              ambiguous, rewrites left ─► rewriter ─► retriever (bounded by CRAG_MAX_REWRITES)
              otherwise ─► web search (Tavily, results graded) ─► generator
              no usable evidence ─► fixed "not found" answer (no LLM call)
```

* **Evidence classes.** Paper passages and web results are kept apart: web results are labelled
  `🌐 web` in prompts, citations and the UI.
* **Untrusted content.** Paper text and web pages are wrapped in `<source>` tags and every prompt
  tells the model to ignore instructions inside them. No node executes tools or code.
* **Bounded cost.** One grading call per retrieval round, one call per rewrite, one generation call
  and one verification call: at most 8 LLM calls per question with the defaults.
* **Observability.** Every node appends a trace entry (duration and a short detail) that the API
  returns and the UI shows; problems (index missing, grader failure, web search unavailable)
  are returned as user-visible warnings.

## Getting started

Prerequisites: Python **3.10–3.12** (NumPy < 2 has no wheels for 3.13), a
[Gemini API key](https://aistudio.google.com/app/apikey), optionally a [Tavily](https://tavily.com) key.

```bash
git clone https://github.com/Sahvendra7/corrective-RAG-academic-analyser.git
cd corrective-RAG-academic-analyser
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Optional but recommended on machines without a GPU: CPU-only torch (avoids ~GBs of CUDA wheels)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e ".[test]"
cp .env.example .env        # then put your keys in .env
```

## Building the index

The repository ships `data/processed/metadata.json` (480 papers) and the v0.1 text/chunk files,
but not the PDFs. Build the index in one of two ways:

```bash
# A. Full quality: download the exact paper versions listed in metadata.json (~25 min, polite 2 s delay),
#    then parse PDFs (pages + sections) and build the index
python -m src.ingestion.arxiv_downloader --from-metadata
python -m src.vectorstore.faiss_store --build

# B. Quick: no downloads; chunk the committed v0.1 text files (no page numbers or reliable sections)
python -m src.vectorstore.faiss_store --build --text-only
```

`--build` parses every PDF found in `data/raw/` (cached in `data/processed/structured/`), falls back
to the `.txt` file for papers without a usable PDF, embeds all chunks and writes
`data/index/{manifest.json, faiss.index, chunks.jsonl}` atomically. The manifest records the
embedding model, dimension, chunking parameters and a checksum; loading refuses an index whose
files disagree or that was built with a different embedding model. Rebuild after changing
`CRAG_EMBEDDING_MODEL` or chunking settings. `python -m src.vectorstore.faiss_store` (no flag)
prints index statistics.

To add papers, run `python -m src.ingestion.arxiv_downloader` (searches the queries in that file and
appends to `metadata.json`), then rebuild. Rebuilding is a full rebuild by design (~10 min on CPU for
~25k chunks); incremental appends were the source of stale/misaligned indexes in v0.1.

## Running the API and UI

```bash
uvicorn src.api.server:app --port 8000      # API docs at http://localhost:8000/docs
streamlit run ui/app.py                      # UI at http://localhost:8501 (CRAG_API_URL to point elsewhere)
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /ready` | 200 when the index is loaded and `GEMINI_API_KEY` is set, else 503 with the reason |
| `GET /papers` | indexed papers (used by the UI's paper filter) |
| `POST /chat` | `{"query", "paper_ids"?, "allow_web_search"?}` → JSON result |
| `POST /chat/stream` | same input; SSE `node_update`* → `result` → `done` (or `error` → `done`); `: keepalive` comments every 15 s |

The `result` payload contains `answer_text`, `citations` (paper/web, title, arXiv id, section,
pages, URL), `evidence` (all passages given to the generator, with grade, similarity and whether
cited), `grounding` (`unsupported_claims`, `unsupported_numbers`, `invalid_citations`, `checked`),
`warnings`, `trace`, `llm_calls`, `retry_count` and `web_search_used`.

Limits: query length (`CRAG_MAX_QUERY_CHARS`), ≤50 paper ids, 64 KB request body, a per-client rate
limit (`CRAG_RATE_LIMIT`, in-memory per process), a concurrency cap (`CRAG_MAX_CONCURRENT_REQUESTS`,
excess requests get 503 + `Retry-After`) and a per-request deadline (`CRAG_REQUEST_TIMEOUT_S`).
Error responses never include exception text.

## Configuration

All settings live in `src/config.py` and are read from the environment or `.env` as `CRAG_<NAME>`
(API keys use their usual names).

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | – | required for grading, generation and grounding checks |
| `TAVILY_API_KEY` | – | enables the web-search fallback |
| `CRAG_LLM_MODEL` | `gemini-3.1-flash-lite` | Gemini model |
| `CRAG_LLM_TIMEOUT_S` / `CRAG_LLM_MAX_RETRIES` | 60 / 2 | per-call timeout and retries on transient errors |
| `CRAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | `hashing` = offline test embedder |
| `CRAG_CHUNK_MAX_WORDS` / `_OVERLAP_WORDS` / `_MIN_WORDS` | 170 / 30 / 25 | chunk sizing (fits MiniLM's 256-token window) |
| `CRAG_INDEX_REFERENCES` | false | index bibliography sections |
| `CRAG_CONTEXTUAL_HEADERS` | true | embed "title \| section" with each chunk |
| `CRAG_TOP_K` | 5 | passages retrieved per round |
| `CRAG_MAX_REWRITES` | 2 | query rewrites before falling back to the web (0–5) |
| `CRAG_WEB_SEARCH_ENABLED` | true | global switch for Tavily |
| `CRAG_WEB_MAX_RESULTS` / `CRAG_WEB_TIMEOUT_S` | 3 / 15 | Tavily request |
| `CRAG_MAX_EVIDENCE_DOCS` / `CRAG_MAX_EVIDENCE_CHARS` | 6 / 1500 | evidence given to the generator and verifier |
| `CRAG_VERIFY_GROUNDING` | true | run the LLM grounding check |
| `CRAG_CORS_ORIGINS` | `["http://localhost:8501"]` | JSON list |
| `CRAG_RATE_LIMIT` | `20/minute` | per client IP |
| `CRAG_REQUEST_TIMEOUT_S` | 180 | per request |
| `CRAG_MAX_CONCURRENT_REQUESTS` | 4 | pipeline runs in parallel |
| `CRAG_INDEX_DIR` | `data/index` | index location |

Queries are logged at INFO level (truncated). LangSmith tracing is off unless you enable it in `.env`.

## Testing

```bash
pytest                                  # hermetic: no network, keys or model downloads (~10 s)
pytest -m network tests/test_pipeline.py  # downloads the CRAG paper + MiniLM; real parse → index → search
pytest -m llm                           # tests that call Gemini (need GEMINI_API_KEY)
```

The default suite runs the real LangGraph graph over a real FAISS index with a deterministic
hashing embedder, a fake LLM at the LangChain boundary and a fake Tavily client; PDFs are generated
with PyMuPDF. It covers parsing (columns, headers, hyphenation, run-in headings, malformed and
encrypted files), chunking, index consistency and corruption detection, routing and loop
termination, failure paths, citation validation, the API/SSE contract and the Streamlit app
(`streamlit.testing`).

## Evaluation

**Offline retrieval benchmark** (no API key; needs the embedding model):

```bash
python -m src.evaluation.retrieval_benchmark --build     # builds data/benchmark/*, writes reports/retrieval_benchmark.json
```

It runs the 51 questions in `data/processed/eval_dataset.json` against several index configurations
with the same embedding model and reports paper hit@k, MRR, evidence recall@k (share of the gold
span's word 5-grams present in the passages *as delivered to the generator*), words delivered,
bibliography share and search latency. Metric definitions and known biases are in the module
docstring; current results are in [`reports/retrieval_benchmark.md`](reports/retrieval_benchmark.md).

**LLM-judged answer evaluation** (needs `GEMINI_API_KEY`): `src/evaluation/run_eval.py`,
`graph_naive.py` (naive-RAG baseline using the same index and `top_k`) and
`ragas_only_evaluation.py` (RAGAS faithfulness / answer relevancy / context precision over the
saved CSVs). The CSVs in `data/processed/` were produced by the v0.1 pipeline; they have not been
regenerated for the current pipeline.

## Project layout

```text
src/
  config.py               settings (pydantic-settings) + v0.1 constant aliases
  ingestion/              arxiv_downloader, pdf_parser, sections, chunker
  vectorstore/            embeddings (MiniLM / hashing), faiss_store (build, load, search)
  pipeline/               state, graph, llm_factory, evidence (prompt delimiting, citations,
                          numeric check), nodes/ (retriever, grader, rewriter, web_search,
                          generator, hallucination)
  api/server.py           FastAPI app + SSE
  evaluation/             retrieval_benchmark (offline), run_eval / graph_naive / ragas (LLM)
ui/app.py                 Streamlit UI
tests/                    hermetic suite (+ opt-in network/LLM tests)
reports/                  benchmark results
docs/AUDIT.md             audit findings and their status
data/processed/           metadata.json, v0.1 texts/chunks, eval set (data/raw, data/index are git-ignored)
```

## Known limitations

* **Section labels are heuristic.** Unrecognised top-level headings are treated as subsections of
  the previous section; a few papers have no detectable headings. Section labels are used for
  display and for excluding bibliography/front matter, not for ranking.
* **Bibliography detection** found a references section in ~81% of the parsed papers; in the rest,
  reference lists can still be indexed as body text.
* **No OCR.** Scanned/image-only PDFs are rejected; such papers fall back to their `.txt` file if one exists.
* **De-hyphenation** cannot tell `large-` + `scale` (compound) from `abili-` + `ties`; the hyphen of a
  compound split at its only hyphen is dropped.
* **Dense retrieval only** (MiniLM, 384-d). Exact identifiers and numbers are matched only as well as
  the embedding model handles them; hybrid/lexical retrieval was not added because it has not been
  evaluated here.
* **Stateless conversations.** Each question is answered independently.
* **Rate limiting and the concurrency cap are per process**; run behind a shared limiter if you
  start several workers.
* On client disconnect or timeout the stream stops and no further pipeline step starts, but an LLM
  call already in flight completes in its worker thread.
* Grading, generation and verification quality with the current prompts has **not** been re-measured
  with Gemini in this revision (no API key was available); see `docs/AUDIT.md`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/ready` → `index not built`, UI warning | build the index (see above) |
| `IndexConsistencyError` on startup | the index files disagree or were built with another embedding model: rebuild |
| Answer says the language model is not configured | set `GEMINI_API_KEY` in `.env` and restart |
| "Web search fallback is unavailable" warning | set `TAVILY_API_KEY`, or ignore (answers use the papers only) |
| `pip install` fails building NumPy | use Python 3.10–3.12 |
| 503 "Server busy" | raise `CRAG_MAX_CONCURRENT_REQUESTS` or retry |

## License

This project is licensed under the MIT License.
