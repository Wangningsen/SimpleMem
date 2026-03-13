# LongHealthMem Evaluation in SimpleMem

This repository now supports a benchmark mode for LongHealthMem in addition to the existing LoCoMo pipeline.

## 1) Runner

Use the unified runner:

```python
python test_benchmark.py --benchmark longhealthmem --dataset benchmark_longmemory_v1.json
```

LoCoMo is still available:

```python
python test_benchmark.py --benchmark locomo --dataset test_ref/locomo10.json
```

## 2) Expected LongHealthMem Input Schema

The LongHealthMem adapter expects a patient-id keyed JSON object:

```json
{
  "patient_01": {
    "name": "...",
    "birthday": "...",
    "diagnosis": "...",
    "texts": {
      "text_0": {
        "text": "long clinical text",
        "global_start": 0,
        "global_end": 31720
      },
      "text_1": {
        "text": "follow-up text",
        "global_start": 31720,
        "global_end": 44556
      }
    },
    "questions": [
      {
        "No": 0,
        "question": "...",
        "answer_a": "...",
        "answer_b": "...",
        "answer_c": "...",
        "answer_d": "...",
        "answer_e": "...",
        "correct": "...",
        "correct_letter": "D",
        "ambiguous_correct": false
      }
    ]
  }
}
```

Adapter file: `benchmarks/longhealthmem_adapter.py`

## 3) Patient Text -> Sequential User-Only Turns

LongHealthMem has one information source (the patient record text stream), so each patient's texts are converted into synthetic user-only dialogue turns.

- Texts are processed in deterministic order (`text_0`, `text_1`, ...).
- Default behavior is **one input text = one user turn**.
- Optional fallback chunking is used only when `LONGHEALTHMEM_CHUNK_SIZE > 0` (or `--chunk-size > 0`) and a single text exceeds that limit.
- Each turn is ingested incrementally, and memory is updated after each turn.
- Runtime state is reset between patients so memory never leaks across patient sessions.

Config options (`config.py`):

- `LONGHEALTHMEM_CHUNK_SIZE` (default `0`, disabled)
- `LONGHEALTHMEM_CHUNK_OVERLAP` (default `0`)
- `LONGHEALTHMEM_USER_SPEAKER`

CLI overrides (fallback chunking example):

```python
python test_benchmark.py --benchmark longhealthmem --chunk-size 12000 --chunk-overlap 200
```

## 4) MCQ Prompting and Parsing

LongHealthMem uses closed-ended MCQ answering.

- Prompt utility: `benchmarks/mcq.py::build_mcq_prompt`
- Parser utility: `benchmarks/mcq.py::parse_mcq_choice`

The prompt requires a strict output line:

```text
Final answer: <LETTER>
```

The parser accepts common variants and normalizes to the option letter:

- `A`
- `Option A`
- `(A)`
- `Final answer: A`

## 5) Metrics

For LongHealthMem, evaluation is exact-match MCQ accuracy only (no BLEU/ROUGE/BERTScore/LLM-judge).

Reported fields include:

- `num_questions`
- `num_correct`
- `accuracy`
- `per_patient_accuracy`
- `num_ambiguous_questions`
- `ambiguous_accuracy`

Implementation: `benchmarks/longhealthmem_tester.py`

## 6) Embedding Backend Refactor (Local or API)

`utils/embedding.py` now supports two backends via a provider abstraction:

- Local SentenceTransformer backend (`EMBEDDING_BACKEND="local"`)
- OpenAI-compatible API backend (`EMBEDDING_BACKEND="api"`)

### Required API vars/settings

- `EMBEDDING_API_KEY` (or `DASHSCOPE_API_KEY` as fallback)
- `EMBEDDING_API_BASE`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSION`

These can be set in `config.py` or environment variables. Environment variables take precedence.

Implementation notes:

- OpenAI-compatible requests use `dimensions=<int>` when sending embedding requests.
- For DashScope-compatible `text-embedding-v4`, per-request batch size is capped to 10 inputs.
- Query-specific native SDK parameters (for example `text_type="query"`) are not faked in the OpenAI-compatible path.

Example (OpenAI-compatible Qwen endpoint):

```python
EMBEDDING_BACKEND = "api"
EMBEDDING_API_KEY = "your-key"
EMBEDDING_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSION = 1024
```

## 7) Notes

- LoCoMo legacy script (`test_locomo10.py`) remains unchanged for backward compatibility.
- Unified benchmark entrypoint is `test_benchmark.py`.
- No training/evaluation/install is run automatically by these code changes.

