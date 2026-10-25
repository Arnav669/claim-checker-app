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

Edit `.env` and set `GROQ_API_KEY`. Check the current model id on the Groq console models page and set `GROQ_MODEL` if the default is not listed. LangFuse keys are optional; without them tracing is switched off.

Optional providers: `pip install -r requirements-optional.txt`, then set `LLM_PROVIDER=gemini` or `LLM_PROVIDER=claude` with the matching API key.

## Run

```
python scripts/download_data.py
python scripts/build_index.py --backend both
streamlit run app.py
```

Terminal chat instead of the web UI: `python scripts/chat_cli.py`

## Evaluation

```
python scripts/run_eval.py calibrate
python scripts/run_eval.py retrieval
python scripts/run_eval.py backends
python scripts/run_eval.py verdict --per-label 35
python scripts/run_eval.py verdict --per-label 35 --no-rerank
python scripts/run_eval.py ood
```

`calibrate` prints the top rerank score distribution for each label and for out-of-domain claims, and suggests a `GATE_MIN_SCORE`. Put that value in `.env` before running `verdict` and `ood`. Results are written to `reports/` as CSV, JSON and Markdown tables. LLM replies are cached on disk, so repeating a run does not use API quota again.

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
| `GATE_MIN_SCORE` | 0.1 | Minimum rerank score before the LLM is asked for a verdict |
| `MAX_RETRIES` | 1 | Query rewrites allowed before abstaining |
| `LLM_PROVIDER` | `groq` | `groq`, `gemini` or `claude` |
| `LLM_MAX_RPM` | 25 | Client-side request rate cap |
