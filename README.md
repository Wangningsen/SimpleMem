# SimpleMem LongHealthMem Evaluation Guide

This repository is used to run **LongHealthMem** evaluation with the SimpleMem pipeline.

## Scope

- Benchmark target: **LongHealthMem** (patient-level sessions, MCQ evaluation)
- Dataset file: `benchmark_longmemory_v1.json`
- Metric: exact-match MCQ accuracy
- LoCoMo/MCP/general product documentation is intentionally omitted from this README.

## Behavior Summary (Current Implementation)

- One patient is processed as one independent session.
- Default ingestion is **one patient text = one user turn**.
- Optional fallback chunking is available only when explicitly enabled.
- Memory is updated incrementally after each turn.
- Runtime state is reset between patients (vector DB + memory builder state).

## Environment

- OS: Windows (PowerShell examples below)
- Python: use your local interpreter/venv

## 1) Prepare Config

Create a `config.py` from `config.py.example` (or rely on env vars).

Minimum fields you should verify in `config.py`:

```python
# LLM
OPENAI_API_KEY = "..."
OPENAI_BASE_URL = None  # or compatible endpoint
LLM_MODEL = "..."

# Embedding
EMBEDDING_BACKEND = "api"  # or "local"
EMBEDDING_MODEL = "..."
EMBEDDING_DIMENSION = 1024
EMBEDDING_API_KEY = "..."  # DASHSCOPE_API_KEY env fallback is also supported
EMBEDDING_API_BASE = "..."  # optional if default endpoint is desired

# LongHealthMem defaults
LONGHEALTHMEM_CHUNK_SIZE = 0
LONGHEALTHMEM_CHUNK_OVERLAP = 0
LONGHEALTHMEM_USER_SPEAKER = "patient"
```

## 2) Dataset Location

Place or keep the dataset at project root:

- `benchmark_longmemory_v1.json`

## 3) Run LongHealthMem Evaluation

Run full dataset:

```powershell
python test_benchmark.py --benchmark longhealthmem --dataset benchmark_longmemory_v1.json
```

Run partial samples:

```powershell
python test_benchmark.py --benchmark longhealthmem --dataset benchmark_longmemory_v1.json --num-samples 5
```

Specify output file:

```powershell
python test_benchmark.py --benchmark longhealthmem --dataset benchmark_longmemory_v1.json --result-file longhealthmem_test_results.json
```

## 4) Optional Fallback Chunking

Default is no chunking (`--chunk-size 0`).

Enable only if a single text is too long for your model context:

```powershell
python test_benchmark.py --benchmark longhealthmem --dataset benchmark_longmemory_v1.json --chunk-size 12000 --chunk-overlap 200
```

Validation rules:

- `--chunk-size >= 0`
- `--chunk-overlap >= 0`
- if `--chunk-size > 0`, then `--chunk-overlap < --chunk-size`

## 5) Output

Result JSON contains:

- benchmark metadata
- chunking config used
- summary (`num_questions`, `num_correct`, `accuracy`, per-patient stats)
- detailed per-question records

Default output path:

- `longhealthmem_test_results.json`

## 6) Runtime Visibility

Current visibility is terminal log output only, including:

- dataset loading
- per-patient processing
- per-question retrieval/answer progress
- final accuracy summary

No dedicated visualization dashboard is included in this benchmark runner.

## 7) Notes on Timeouts and Retries

- LLM calls currently use retry logic (`max_retries`, exponential backoff).
- There is currently no explicit configurable timeout in `utils/llm_client.py` for chat completions.
- Embedding API path has timeout support (`EMBEDDING_API_TIMEOUT`).
