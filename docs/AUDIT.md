# Audit — corrective-RAG-academic-analyser

This document has two parts:

* **Part 1 — Status (revision 2).** What was verified, fixed, measured and what remains, with the
  evidence for each claim. Written after the changes on branch `claude/zen-galileo-9q2krz`.
* **Part 2 — Original audit** (pre-refactor, commit `7a6918a`), kept verbatim apart from heading
  levels. Its "target architecture" and "file-by-file plan" were proposals; Part 1 says what was
  actually built.

---

## Part 1 — Status (revision 2)

### 1.1 Starting point of this revision

Commit `f3f5bd2` had added `src/config.py` (pydantic-settings), `src/ingestion/sections.py` and a
structure-aware `parse_pdf`. Verification showed that **none of it was wired in**: the chunker still
read the v0.1 flat `.txt` files with the sentence regex, no node read `get_settings()`, and
`Settings` advertised features that did not exist (hybrid retrieval, cross-encoder reranking,
evaluator thresholds "calibrated" in a `reports/retrieval_benchmark.md` that did not exist).
`pip install -e .` did not produce an importable API (`slowapi` and `pydantic-settings` undeclared).

Baseline tests (`pytest tests/`, Python 3.12, no API keys): **10 passed, 2 skipped**. Of the passes,
`test_chunker.py` collected no tests, `test_extraction.py` tested an inline copy of the code,
`test_faiss_store.py` and `test_pipeline.py::test_step6` returned early (passed) when artefacts or
keys were missing, and `test_pipeline.py` downloaded from arXiv and **overwrote tracked files** in
`data/processed/` (run in a scratch worktree to confirm). Only the two latency tests exercised
project code meaningfully.

### 1.2 Status of the original critical findings

| # | Finding | Status | Evidence |
|---|---|---|---|
| C1 | Chunks (mean 535 words) exceed MiniLM's 256-token window | **Fixed** | `chunker.py`: ≤170-word, section-bounded windows; `tests/test_chunker.py`; benchmark below |
| C2 | CRAG worse than naive RAG in the project's own eval | **Not re-measured** | needs `GEMINI_API_KEY`; the mechanisms identified as causes (per-doc grader rejecting local evidence, ungraded web results, 1,200 vs 800 char contexts) were changed — see 1.3 |
| C3 | ~25% of chunks are bibliography | **Mostly fixed** | references excluded at parse level; bibliography share in the top 5 drops from 0.184 to 0.071 (benchmark). A references section was detected in 397 of 479 parsed papers (83%) |
| C4 | Page numbers / sections discarded | **Fixed** | chunks carry `page_start/page_end/section/heading`; citations render `(arXiv:id; Section, p. N)` |
| C5 | Citations written by the LLM | **Fixed** | model emits `[n]` only; invalid markers removed; Sources rendered from metadata (`tests/test_evidence.py`, `test_graph.py::test_fabricated_citations_are_removed`) |
| C6 | Verifier saw less context than the generator; footer was fact-checked | **Fixed** | one `format_evidence()` for grader, generator and verifier; verifier checks `answer_text` without the Sources block |
| C7 | `chunk_all_papers` crashed (`save_metadata` without path) | **Fixed** (function replaced by `chunk_corpus`) | `python -m src.vectorstore.faiss_store --build` ran over all 480 papers |
| C8 | `slowapi` (and `pydantic-settings`) missing from `pyproject.toml` | **Fixed** | clean Python 3.11 venv: `pip install -e ".[test]"` + `pytest` (see 1.5) |
| C9 | Tests not hermetic / testing copies | **Fixed** | 117 hermetic tests; network/LLM tests opt-in via markers and write only to `tmp_path` |

### 1.3 Other changes, by area (each with a regression test)

**PDF extraction** (`pdf_parser.py`) — validated on all 480 corpus PDFs (re-downloaded at the
versions recorded in `metadata.json`, 0 download failures): 479 parse, 1 is rejected by the 50 MB
size guard and falls back to its text file. Parse time on the first 395 papers: p50 211 ms,
p95 2.1 s, max 19.9 s. Defects found on real papers and fixed (each with a regression test):
* Run-in headings ("I." / "INTRODUCTION" / body in one block; IEEE drop caps "R" + "ECENT") were
  missed, leaving whole papers labelled front matter (not indexed) or abstract.
* Paper titles containing section keywords ("…Reasoning **Benchmarks**…") opened sections.
* Lettered appendix headings after the bibliography ("A Additional Experiments") were filed under
  experiments.
* A 7 pt lowercase figure label "references" opened the bibliography mid-paper (arXiv 1606.05112
  lost ~3,000 body words). Keyword heading shortcuts now require an initial capital and text not
  much smaller than body text. A first version of this guard used a 0.9× size floor and rejected
  small-caps headings (8.1 pt "INTRODUCTION" in a 10 pt body), raising mislabelled abstract
  paragraphs from 1,883 to 11,476 across the corpus; the floor is now 0.75× (commit `94d361f`).
* De-hyphenation joined compounds (`state-of-` + `the-art` → `state-ofthe-art`).
* Figure fragments ("1-3", "✦") were kept as paragraphs; a first fix dropped non-Latin text
  (a Korean paper) and was corrected to use Unicode letter classes.
* A document whose text never left the front matter would have produced zero chunks.
* Cached parses had no version, so a rebuild after a parser fix silently reused stale parses;
  the cache now records `PARSER_VERSION` (also stored in the index manifest).

Corpus-wide comparison, parser at `f3f5bd2` vs final parser (all 480 PDFs):

| | `f3f5bd2` | final |
|---|---|---|
| Body words kept (indexable sections) | 3,300,654 | 3,381,110 |
| Words in excluded sections (front matter, references, acknowledgements) | 863,894 | 756,816 |
| Paragraphs labelled "abstract" (mostly mislabelled body text) | 3,276 | 1,883 |

Some papers keep fewer body words than before. Spot checks of 10 of them attribute the difference to
bibliographies that are now detected, garbled equation glyphs (Unicode private-use characters)
that are now dropped, and run-in headings now split from their paragraph — no body text loss was
found in the checked papers after the figure-label fix.

**Index** (`faiss_store.py`, `embeddings.py`) — one directory with a manifest (embedding model,
dimension, counts, chunking parameters, checksum), written atomically and validated on load
(row-count, dimension, checksum and embedder mismatches raise `IndexConsistencyError`). Exact
pre-ranking paper/year/author filters via FAISS ID selectors; de-duplication; `exclude_ids` so
retries fetch new passages. The v0.1 append-only embedding script (which skipped papers by chunk-id
prefix and could not notice re-chunked papers) was replaced by a full rebuild.

**Corrective loop** (`graph.py`, `nodes/*`)
* Grading is one batched structured call per round instead of one call per passage; the overall
  grade covers all passages graded so far; only *relevant* passages are used when any exist.
* Retries no longer re-append duplicate passages; the loop is bounded by `CRAG_MAX_REWRITES`
  (0–5) with an explicit `recursion_limit`; tested for 0, 1, 3 and 5 rewrites.
* The fake "HyDE" step (generated passage never embedded) was removed: one call per rewrite.
* Web results are graded and irrelevant ones dropped; missing key / disabled / failed web search is a
  visible warning. If web search is unavailable, an *irrelevant* retrieval tries rewriting first.
* No usable evidence → fixed abstention message, no generation call.
* Grader, generator and verifier failures are visible (`warnings`, `grounding.checked=false`) instead
  of silently becoming "ambiguous" / "hallucinated"; the sticky `error` field no longer short-circuits
  routing.
* Deterministic `unsupported_numbers` check alongside the LLM claim check.
* Prompts wrap passages in `<source>` tags with an instruction to ignore embedded instructions;
  tag look-alikes inside passages are neutralised.
* LLM client: model/timeout/retries from settings (`max_retries` was 0); `override_llms()` seam
  for tests; clear `LLMNotConfiguredError` when the key is missing.

**API** (`server.py`) — app factory with lifespan initialisation; `/ready` and `/papers`;
non-streaming `/chat`; SSE with `node_update` → `result` → `done` / `error` → `done`, keepalives,
per-request deadline, concurrency cap (503 + Retry-After), configurable rate limit and CORS,
64 KB body limit, request validation (blank/too-long queries, paper-id format, ≤50 ids), generic
error messages (tested not to leak exception text). The unused `force_web_search` field was
replaced by `allow_web_search`; `thread_id` is accepted but ignored (there is no conversation memory).

**UI** (`ui/app.py`) — readiness from `/ready` (v0.1 pinged `/`, a 404, and treated any response as
online), paper filter, web toggle (disabled when the server has no Tavily key), step status,
answer with Sources (📄 paper / 🌐 web, section, pages, link), evidence passages with grade and
similarity, grounding status, warnings and the step trace. Untrusted metadata is Markdown-escaped,
only http(s) links are rendered, and no server text is rendered as HTML.

**Configuration / packaging** — `Settings` trimmed to settings that are read; `state.MAX_RETRIES`
is an alias of `Settings.max_rewrites` (v0.1 had 3, 2 and a hard-coded `TOP_K = 2` in different
files); `requires-python >=3.10,<3.13` (NumPy < 2 has no 3.13 wheels — verified);
`.env.example` no longer lists unused keys or enables LangSmith tracing; CI installs CPU-only torch,
runs a ruff gate for undefined names/syntax errors, and the hermetic suite; `--from-metadata`
downloader mode re-fetches the exact paper versions in `metadata.json`.

### 1.4 Retrieval benchmark

`python -m src.evaluation.retrieval_benchmark --build` → [`reports/retrieval_benchmark.md`](../reports/retrieval_benchmark.md)
(per-question data in `.json`). 51 questions from `data/processed/eval_dataset.json`, same
embedding model (all-MiniLM-L6-v2) for every configuration, no LLM involved. Evidence recall is
measured on the text *as delivered to the generator* (v0.1 truncated passages to 1,200 characters).

| Metric | v0.1 (baseline) | new chunker on v0.1 text | PDF parse + headers | **PDF parse (default)** |
|---|---|---|---|---|
| Paper hit@1 | 0.608 | 0.608 | 0.667 | **0.745** |
| Paper hit@5 | 0.804 | 0.686 | 0.726 | **0.824** |
| Paper MRR@10 | 0.690 | 0.645 | 0.696 | **0.778** |
| Evidence recall@2 | 0.199 | 0.321 | 0.362 | **0.422** |
| Evidence recall@5 | 0.299 | 0.470 | 0.477 | **0.504** |
| Words delivered @5 | 928 | 702 | 669 | **673** |
| Bibliography share @5 | 0.184 | 0.047 | 0.051 | **0.071** |

Paired per-question comparison, default vs v0.1: evidence recall@2 better on 22 questions, worse
on 6, tied on 23; evidence recall@5 24/8/19; paper reciprocal rank 11/8/32. At v0.1's operating
point (top-k = 2) evidence recall rises from 0.199 to 0.422 with fewer words delivered; with the
new default top-k = 5 it is 0.504.

Decisions taken from these measurements:
* **Contextual headers off by default.** Embedding "title | section" with each chunk was assumed
  to help; it did not (default vs headers: paper RR better on 8 questions, worse on 3; evidence
  recall@2 10/6). The effect is small on 51 questions, so the conclusion is "no evidence it helps".
* **No per-paper diversity cap.** The new-chunker-on-text index loses paper hit@5 (0.804 → 0.686)
  because one similar paper can fill the top 5 (mean 2.2 distinct papers). A per-paper cap was
  tried: on that index it raised paper hit@5 only to 0.706 (cap 2) while evidence recall@5 fell
  from 0.470 to 0.355, so it was not adopted. The structured-PDF index does not show the paper-hit
  loss.

Caveats: the questions were generated by Gemini from randomly sampled v0.1 chunks (so every gold
span lies inside one v0.1 chunk — an advantage for the baseline on paper hit), are single-paper and
extractive, and include no unanswerable or multi-paper questions. Search latency (MiniLM query
embedding + flat FAISS search, CPU) was 10–40 ms in all configurations and varied by a factor of
~1.7 between runs on this shared machine; it is not a basis for comparison.

### 1.5 Verification performed

| Check | Result |
|---|---|
| `pytest` (hermetic, Python 3.12 dev venv) | 117 passed, 5 deselected (network/LLM) |
| Clean Python 3.11 venv following README (`pip install torch --index-url …/cpu`, `pip install -e ".[test]"`, `pip check`, `pytest`) | no broken requirements; 117 passed, 5 deselected |
| `pytest -m network tests/test_pipeline.py` (real CRAG paper from arXiv, real MiniLM) | 2 passed, 1 skipped (Gemini step: no key) |
| `ruff check --select F821,F811,E9 src ui tests` | clean |
| Parser over 480 real PDFs | 479 parsed, 1 rejected (size guard) |
| Streamlit UI driven in headless Chromium against the real 480-paper index (fake LLM for answer text) | sources with section/pages, evidence, warnings and trace render; invalid citation removed |
| `pytest -m llm` / end-to-end answer quality with Gemini | **not run** — no `GEMINI_API_KEY` in this environment |
| Web fallback against the real Tavily API | **not run** — no `TAVILY_API_KEY` (covered by a fake client in tests) |

### 1.6 Remaining issues and limitations (by severity)

**High**
* End-to-end answer quality (faithfulness, relevancy, citation precision) of the revised pipeline is
  unmeasured; the v0.1 CSVs in `data/processed/` describe the old pipeline. Batched grading and the
  new prompts are plausible improvements but unverified with Gemini.
* The question set is LLM-generated from v0.1 chunks: single-paper, extractive, no unanswerable,
  comparative, multi-paper or adversarial questions. A hand-labelled set is needed for claims
  beyond retrieval.

**Medium**
* Section classification is keyword-based; unrecognised top-level headings become subsections of
  the previous section. 17% of papers have no detected bibliography, so some reference lists are
  still indexed (bibliography share in the top 5 is 0.071).
* Dense-only retrieval; lexical/hybrid retrieval and reranking were not evaluated, so not added.
* Rate limiting and the concurrency cap are per process; a node already running (LLM call) is not
  cancelled on client disconnect/timeout.
* No OCR for scanned PDFs.

**Low**
* Compound words split at their only hyphen lose the hyphen.
* `langgraph` emits a `LangChainPendingDeprecationWarning` (checkpoint serializer default) — a
  future-compatibility item when upgrading LangGraph.
* Legacy evaluation scripts (`evaluation.py`, `run_eval.py`) evaluate questions about this codebase
  rather than the corpus; they were left unchanged apart from the naive baseline's fairness fix
  (same index and `top_k` as CRAG).

---

## Part 2 — Original audit (pre-refactor, commit `7a6918a`)

This audit was written **before** any code was changed. Every finding below was
verified by reading the code or running it; numbers come from the repository's
own data files.

---

### 1. Architecture at the time

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

### 2. Critical problems

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

### 3. Other weaknesses by category

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

### 4. Improvements, prioritised (original proposal)

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

### 5. Target architecture (original proposal — only partly adopted, see Part 1)

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

### 6. File-by-file plan (original proposal — only partly adopted, see Part 1)

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
