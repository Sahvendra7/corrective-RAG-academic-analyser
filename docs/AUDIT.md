# Audit Report — corrective-RAG-academic-analyser (pre-refactor, commit `7a6918a`)

This audit was written **before** any code was changed. Every finding below was
verified by reading the code or running it; numbers come from the repository's
own data files.

---

## 1. Current architecture

```
arxiv_downloader.py ─► data/raw/*.pdf
pdf_parser.py       ─► data/processed/texts/*.txt   (pages concatenated, page numbers lost)
chunker.py          ─► data/processed/chunks/*.json (sentence-accumulated, ~256-word target)
embeddings.py       ─► data/embeddings/{embeddings.npy, chunk_ids.json, chunk_registry.json}
faiss_store.py      ─► data/embeddings/faiss.index  (IndexFlatIP, all-MiniLM-L6-v2)

LangGraph (src/pipeline/graph.py)
START → retriever(top_k=2) → grader(1 LLM call per doc)
        ├─ relevant   → generator → hallucination_checker → END
        ├─ irrelevant → web_search(Tavily) → generator → hallucination_checker → END
        └─ ambiguous  → rewriter(2 LLM calls) → retriever … (≤3×) → web_search → …

FastAPI  : POST /chat/stream (SSE of node names + generation), GET /health
Streamlit: chat UI that shows node names and the final markdown answer
Eval     : run_eval.py (CSV of answers) → ragas_only_evaluation.py (RAGAS w/ Gemini judge)
```

## 2. Critical problems

| # | Problem | Evidence | Why it matters |
|---|---------|----------|----------------|
| C1 | **Most of each chunk is never embedded.** Mean chunk length is **535 words** (target 256) because the sentence regex `(?<=[.!?])\s+(?=[A-Z])` fails on PDF line-wrapped text, tables and reference lists. `all-MiniLM-L6-v2` truncates at 256 word-pieces (~190 words). | `data/processed/chunks/*.json`: 9,875 chunks, mean 535 words | ~65% of the corpus text is invisible to dense retrieval. This is the single largest retrieval-quality defect. |
| C2 | **CRAG performs worse than naive RAG in the project's own results.** | `evaluation_results_crag_*.csv` vs `..._naive_*.csv` (50 Qs): faithfulness 0.918 vs 0.961, answer relevancy 0.675 vs 0.760, context precision 0.482 vs 0.490, latency **30.2 s vs 1.8 s**, web search on **46%** of questions that were generated *from the local corpus* | The corrective mechanism is actively harmful: the per-doc grader rejects good local evidence and routes to the web. |
| C3 | **~25% of chunks are reference-list noise.** | 2,427 / 9,875 chunks contain ≥4 "et al." or "arXiv preprint" | Bibliography text matches almost any query lexically/semantically and pushes real evidence out of top-k. |
| C4 | **Page numbers, sections and paper structure are discarded** at parse time. | `pdf_parser.extract_text_from_pdf` joins pages; chunks carry no page/section | Page-level citations (`[Paper, p. 5]`) are impossible; no section-aware retrieval. |
| C5 | **Citations are produced by the LLM.** The generator prompt asks the model to write a "Sources" section with titles and URLs. | `generator.py` system prompt rule 7 | Citations can be invented; nothing validates `[Doc N]` indices. |
| C6 | **Hallucination checker sees less context than the generator.** Generator uses 1,200 chars/doc, checker 800 chars/doc (max 6,000 total). | `generator.py:build_context`, `hallucination.py:build_check_context` | Supported claims are flagged as hallucinations (false positives); also the generator's markdown footer is fact-checked. |
| C7 | **`chunker.chunk_all_papers` crashes** — `save_metadata(metadata)` is called without the required `file_path`. | `chunker.py` (inside the loop) | Re-chunking the corpus is impossible without editing code. |
| C8 | **`slowapi` and `google-genai` are missing from `pyproject.toml`.** | `pip install .` then `import src.api.server` → `ModuleNotFoundError` | The documented install path (`pip install -e .`) does not produce a runnable API. |
| C9 | **Tests are not hermetic and some test nothing.** `test_pipeline.py` downloads from arXiv and writes into `data/`; `test_chunker.py`/`test_extraction.py` re-implement the code they claim to test; `test_faiss_store.py` silently returns (passes) when artefacts are missing. | running `pytest` created `data/raw/` and `data/embeddings_test/` | CI is network-flaky and gives false confidence. |

## 3. Other weaknesses by category

**Bugs / correctness**
- Retriever appends duplicates on retries (no de-duplication); duplicates reach the generator.
- `relevant_documents` includes *ambiguous* docs, so weakly related text is fed to generation.
- Overall grade is computed only over *new* docs on retries, so the decision ignores evidence already accepted.
- `state.MAX_RETRIES = 3`, `config.MAX_RETRIES = 2`, `retriever.TOP_K = 2`, `config.TOP_K = 3` — conflicting magic numbers.
- `error` is never cleared; once set it short-circuits routing for the rest of the run.
- The "HyDE" passage is never embedded — it is only pasted into the rewrite prompt, so the technique is mislabelled and costs an extra LLM call.
- API passes a `thread_id` but the graph has no checkpointer (no memory, misleading).
- UI health check pings `/` (404) and treats any response as "online".
- `graph_naive.py` builds a FAISS store at import time and rebuilds the graph for every query.

**Architecture / complexity**
- No LLM abstraction → nodes call Gemini directly; untestable without mocking internals.
- Prompts scattered across six modules; no shared rules for untrusted content.
- `__main__` demo blocks in every node with stale state shapes.
- Root-level debug file `test_grader_issue.py`.

**Typing**: `CRAGState` is a TypedDict of loosely typed `dict`s; `document_grades: list[dict]`; grades are free strings.

**Error handling**: `max_retries=0` on the LLM client means any transient 429 fails a node; failures degrade silently into "ambiguous"/"hallucinated" with no trace.

**Security**
- Retrieved PDF text and web pages are concatenated into prompts with no delimiting → prompt-injection surface.
- Web results are added to the evidence set **without grading**.
- CORS `*`, no request-size limit, no concurrency cap, no timeout on graph execution.
- `.env.example` lists unused keys (OpenAI, Cohere) and enables LangSmith tracing by default (sends content to a third party).
- UI renders server error text with `unsafe_allow_html=True`.

**Performance**: N sequential LLM grading calls; up to 3 rewrite loops × (2 rewrite calls + N grading calls); no caching of query embeddings or web results; graph rebuilt per query in naive eval.

**Retrieval**: dense-only, top-k=2, no lexical matching (bad for acronyms, model names, numbers), no reranking, no metadata filters used, no paper-name awareness.

**Evaluation**
- No retrieval metrics at all (Recall@K, MRR, nDCG).
- Golden set: 51 LLM-generated, single-paper, extractive questions; no unanswerable, comparative, numerical or adversarial cases; many say "the proposed framework" without naming the paper.
- Generator and judge are the same model (self-preference bias); naive used top-k=5 while CRAG used top-k=2 (unfair comparison).
- `evaluation.py` evaluates questions about *this codebase*, not the corpus.

**Citation/grounding**: binary hallucination flag, no claim-level analysis, no numeric check, no "insufficient evidence" policy, unsupported claims are appended as a warning but the answer is kept.

**UI/UX**: no sources, pages, scores, correction path, confidence or latency are shown.

**Production**: no structured logs, no request IDs, no readiness probe, no API versioning, no lint/type-check in CI, CI installs GPU-sized torch on three Python versions.

## 4. Improvements, prioritised

**High impact**
1. Structure-aware PDF parsing (pages, headings, de-hyphenation, repeated header/footer removal) and section-aware chunking sized to the embedder's window; drop the bibliography from the index.
2. Hybrid retrieval (BM25 + dense, reciprocal-rank fusion) + optional cross-encoder reranking + de-duplication + metadata filters — each **measured** on a labelled retrieval benchmark.
3. Make the retrieval evaluator explicit: three-way decision (correct / ambiguous / incorrect) with upper/lower thresholds as in the CRAG paper; use a cross-encoder as the lightweight evaluator and the LLM only for the uncertain band (one batched call, not N).
4. Citations generated from evidence metadata only; validated `[n]` markers rendered as `[Title, p. X]`.
5. Claim-level grounding verification with deterministic numeric checks, a bounded regenerate-once policy and an explicit "insufficient evidence" answer.
6. Reproducible evaluation: hand-written golden set with categories and evidence labels; retrieval metrics computed offline; end-to-end ablation harness.

**Medium impact**
7. Typed state (Pydantic models), centralised prompts with untrusted-data delimiting, LLM protocol for testability, loop limits.
8. Query analysis that decides *whether* to rewrite (acronyms, follow-ups, comparisons, paper references) and decomposes comparisons into per-paper retrieval.
9. FastAPI: versioned routes, request/response models, request IDs, structured errors, readiness, timeouts, body-size limit, concurrency cap, configurable CORS and rate limits, typed SSE events.
10. Structured JSON logging + per-request trace (nodes, timings, LLM calls, tokens).
11. Hermetic test suite with a fake LLM and a deterministic embedder; CI with ruff, mypy and pytest.

**Nice to have**
12. Paper summary / comparison / evidence-extraction endpoints.
13. Caching of query embeddings, retrieval results and web searches.
14. Streamlit UI showing evidence, scores, correction path, grounding and latency.

## 5. Target architecture

```
ingestion/   pdf_parser (structured pages+headings) → sections (canonical labels)
             → chunker (section-bounded, token-sized, contextual header) → corpus.jsonl
retrieval/   embedder · dense(FAISS) · bm25 · fusion(RRF) · reranker(cross-encoder)
             · HybridRetriever (filters, dedupe, section prior, caching)
pipeline/    state (typed) · prompts · llm (protocol + Gemini + call accounting)
             · query_analysis · nodes/* · citations · grounding · graph (bounded loops)
research/    paper summary · paper comparison (structured)
api/         app factory · schemas · middleware (request id, size limit) · routes · SSE
evaluation/  golden set · retrieval metrics · retrieval benchmark (offline)
             · end-to-end ablation harness (needs API key) · LLM judge
observability.py  JSON logs + request context + LLM usage accounting
```

Graph:

```
analyze_query → retrieve → evaluate_retrieval
   correct   ───────────────────────────────► generate
   ambiguous → refine_evidence (+ web if enabled) → generate
   incorrect → rewrite_query → retrieve (≤ max_rewrites)
             → web_search (if enabled) → generate
generate → verify_grounding
   grounded / budget exhausted → finalize → END
   ungrounded & budget left    → generate (with verifier feedback, ≤ max_regenerations)
```

## 6. File-by-file plan

| File | Action |
|------|--------|
| `src/config.py` | Replace constants with a `pydantic-settings` `Settings` object; keep old module-level names as aliases. |
| `src/ingestion/pdf_parser.py` | Rewrite: page-aware block extraction, font-size heading detection, de-hyphenation, repeated header/footer removal; keep `clean_text`/`extract_text_from_pdf` API. |
| `src/ingestion/sections.py` | New: map headings to canonical academic sections. |
| `src/ingestion/chunker.py` | Rewrite: section-bounded, token-budgeted chunks with page/section/paragraph metadata; fix C7. |
| `src/ingestion/build_corpus.py` | New CLI: PDFs → structured chunks → indexes. |
| `src/retrieval/*` | New: embedder, dense/BM25 indexes, RRF fusion, reranker, hybrid retriever. |
| `src/vectorstore/faiss_store.py`, `embeddings.py` | Keep as thin backward-compatible wrappers over the new index. |
| `src/pipeline/state.py` | Typed models (`Evidence`, `GradedEvidence`, `GroundingReport`, `TraceEvent`, …). |
| `src/pipeline/llm.py` (+ `llm_factory.py` shim) | LLM protocol, Gemini client with retries/timeouts, usage accounting. |
| `src/pipeline/prompts.py` | All prompts; untrusted-content delimiters. |
| `src/pipeline/query_analysis.py` | Rule-based analysis + acronym lexicon + paper resolution. |
| `src/pipeline/nodes/*` | Rewrite each node around the typed state; add `evaluate_retrieval`, `refine`, `verify`, `finalize`. |
| `src/pipeline/citations.py`, `grounding.py` | Citation parsing/validation/rendering; claim-level verification + numeric checks. |
| `src/pipeline/graph.py` | Bounded graph, `run_query`, `PipelineConfig` variants for ablations. |
| `src/research/*` | Paper summary and comparison services. |
| `src/api/*` | App factory, schemas, middleware, routes, SSE. |
| `src/observability.py` | JSON logging, request context, usage accounting. |
| `src/evaluation/*` | Golden set, metrics, retrieval benchmark, e2e harness, judge; legacy scripts retained. |
| `ui/app.py` | Evidence-first research UI. |
| `tests/*` | Replace with hermetic unit/integration tests. |
| `.github/workflows/ci.yml` | ruff + mypy + pytest(cov) on CPU torch. |
| `README.md`, `.env.example`, `pyproject.toml`, `requirements.txt` | Rewrite / correct. |
