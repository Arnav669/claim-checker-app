# Claim Checker: an evidence-grounded RAG fact-checking agent

A retrieval-augmented agent that checks scientific claims against a corpus of research abstracts (SciFact) and answers with a verdict (SUPPORTED, REFUTED or NOT_ENOUGH_INFO), a short justification, and cited evidence. It abstains when the corpus does not contain relevant evidence, and it supports follow-up questions within a conversation.

Research tool only. It does not give medical advice.

## Architecture

```
claim -> router -> retrieve (Chroma/FAISS, top-15) -> rerank (cross-encoder, top-4) -> confidence gate
                                                                                   |-- low score, first try -> rewrite query -> retrieve again
                                                                                   |-- low score, retries used -> abstain (NOT_ENOUGH_INFO)
                                                                                   |-- ok -> judge (LLM, JSON verdict) -> grounding check -> reply
follow-up question -> router -> answer from the last verdict and its passages
```

The flow is a LangGraph state graph. Each step is a node, so every step shows up as its own span in LangFuse and in the per-stage latency numbers.

## Repository layout

| Path | Responsibility |
|---|---|
| `factcheck/config.py` | All settings, overridable through environment variables |
| `factcheck/schemas.py` | Pydantic models shared by every module |
| `factcheck/data_loader.py`, `factcheck/chunking.py` | Module 1: SciFact loading, cleaning, sentence-aware chunking |
| `factcheck/embeddings.py`, `factcheck/vector_store.py`, `factcheck/retriever.py`, `factcheck/indexing.py` | Module 2: embeddings, Chroma and FAISS behind one interface, retrieval, index building |
| `factcheck/llm.py`, `factcheck/prompts.py`, `factcheck/parsing.py` | Module 3: provider-swappable LLM with cache and rate limiter, prompts, output parsing |
| `factcheck/graph.py`, `factcheck/agent.py`, `factcheck/memory.py` | Module 3: LangGraph flow, session memory, agent facade |
| `factcheck/reranker.py`, `factcheck/observability.py` | Module 4: cross-encoder reranking, LangFuse tracing |
| `evaluation/` | Metrics and runners for retrieval, verdicts, backend comparison, out-of-domain abstention and gate calibration |
| `scripts/` | Data download, index build, evaluation, CLI chat |
| `app.py` | Streamlit chat UI |
| `tests/` | Unit and end-to-end tests that run offline with stand-in components |

Every component is passed in through constructors (`factcheck/factory.py` does the wiring), so any piece can be swapped or replaced in tests.

## Setup

macOS / Linux:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry. The prompt should show `(.venv)` before you run `pip install`; if it does not, the packages are going into your global Python.

If `.venv` ends up with `bin`, `lib` and `include` folders instead of `Scripts`, `python` on your PATH is not the standard Windows Python (for example MSYS2). Delete `.venv` and create it with the launcher instead: `py -3.13 -m venv .venv`. On a slow connection, add `--default-timeout=120 --retries 10` to `pip install`. If the Streamlit launcher fails, use `python -m streamlit run app.py`.

Edit `.env` and set `GROQ_API_KEY`. The default model is `openai/gpt-oss-120b`. Groq retired its Llama 3.x models from the free tier in August 2026, so check the Groq console models page and set `GROQ_MODEL` if the default is no longer listed. The free tier allows about 8,000 tokens per minute and 200,000 per day, which limits how many claims you can judge in a day. LangFuse keys are optional; without them tracing is switched off.

Optional providers: `pip install -r requirements-optional.txt`, then set `LLM_PROVIDER=gemini` or `LLM_PROVIDER=claude` with the matching API key.

## Run

```
python scripts/download_data.py
python scripts/build_index.py --backend both
streamlit run app.py
```

Terminal chat instead of the web UI: `python scripts/chat_cli.py`

## Document Auditor

Besides checking one claim at a time, the app can audit a whole document. It splits the text into sections, asks the LLM to pull out the checkable factual claims, runs each claim through the same retrieve, rerank, gate and judge pipeline, and writes a report with a verdict and cited evidence for every claim.

- Web UI: choose **Audit a document** in the sidebar, upload a PDF, TXT or MD file (or paste text) and press the audit button. The Markdown report can be downloaded.
- Terminal: `python scripts/audit_doc.py path/to/file.pdf --max-claims 8` writes `reports/audit_<name>.md`.

Claims that use strong wording ("always", "proves", "cures") and are not supported by the evidence are flagged as possibly overstated. This is a keyword heuristic, not a model judgement. "Not enough info" means the knowledge base has nothing on the claim, which does not make the claim false: the knowledge base covers biomedical abstracts only.

Each verified claim costs about 2,000 tokens, so the auditor checks at most `AUDIT_MAX_CLAIMS` claims, spread across the sections, and waits `AUDIT_PAUSE_SECONDS` between claims to stay under the Groq per-minute limit.

## Evaluation

```
python scripts/run_eval.py calibrate
python scripts/run_eval.py retrieval
python scripts/run_eval.py backends
python scripts/run_eval.py verdict --per-label 10
python scripts/run_eval.py verdict --per-label 10 --no-rerank
python scripts/run_eval.py ood
```

`calibrate` prints the top rerank score distribution for each label and for out-of-domain claims, and suggests a `GATE_MIN_SCORE`. Put that value in `.env` before running `verdict` and `ood`. Results are written to `reports/` as CSV, JSON and Markdown tables. LLM replies are cached on disk, so repeating a run does not use API quota again.

## Results

Measured on the SciFact dev split with the default settings (`GATE_MIN_SCORE=0.12`, Groq `openai/gpt-oss-120b`).

| Test | Result |
|---|---|
| Retrieval, 188 claims with gold evidence | hit@5 0.915 dense, 0.942 reranked; MRR 0.839 dense, 0.849 reranked |
| Chroma vs FAISS | Same hit@5 (0.904 on top-5 chunks); FAISS 0.55 ms vs Chroma 3.49 ms per query; 99.5% result overlap |
| Verdicts, 30 claims (10 per label) | Accuracy 86.7%, macro-F1 0.862 |
| Out-of-domain claims, 12 | 12 of 12 refused (abstain rate 1.0) |
| Gate calibration | Out-of-domain scores at most 0.001, real claims 5th percentile 0.17 or higher; suggested threshold 0.1236 |

The verdict sample is small. Treat the accuracy figure as an indication, not a tight estimate.

## Tests

```
pytest
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | 200 / 40 | Chunk size and overlap, in words |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence embedding model |
| `VECTOR_BACKEND` | `chroma` | `chroma` or `faiss` |
| `RETRIEVAL_TOP_K` / `RERANK_TOP_N` | 15 / 4 | Candidates retrieved and passages kept after reranking |
| `GATE_MIN_SCORE` | 0.12 | Minimum rerank score before the LLM is asked for a verdict |
| `MAX_RETRIES` | 1 | Query rewrites allowed before abstaining |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Model id used when the provider is Groq |
| `LLM_PROVIDER` | `groq` | `groq`, `gemini` or `claude` |
| `LLM_MAX_RPM` | 25 | Client-side request rate cap |
| `AUDIT_MAX_CLAIMS` | 8 | Claims checked per document audit |
| `AUDIT_SECTION_WORDS` | 300 | Section size, in words, for claim extraction |
| `AUDIT_MAX_SECTIONS` | 10 | Sections read per document |
| `AUDIT_PAUSE_SECONDS` | 8 | Minimum seconds between claim checks (rate-limit guard) |
