"""
Runtime config loader with safe defaults.

This module avoids hard import failures when `config.py` is not present.
Resolution order for each setting is:
1) hardcoded default
2) user `config.py` value (if file exists and loads)
3) environment variable override
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional


_ROOT_DIR = Path(__file__).resolve().parents[1]
_USER_CONFIG_PATH = _ROOT_DIR / "config.py"


_DEFAULTS: Dict[str, Any] = {
    # LLM
    "OPENAI_API_KEY": None,
    "OPENAI_BASE_URL": None,
    "LLM_MODEL": "gpt-4.1-mini",
    "ENABLE_THINKING": False,
    "USE_STREAMING": True,
    "USE_JSON_FORMAT": False,
    # Embedding
    "EMBEDDING_BACKEND": "local",
    "EMBEDDING_PROVIDER": "local",
    "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
    "EMBEDDING_DIMENSION": 1024,
    "EMBEDDING_CONTEXT_LENGTH": 32768,
    "EMBEDDING_API_KEY": None,
    "EMBEDDING_API_BASE": None,
    "EMBEDDING_API_BATCH_SIZE": 64,
    "EMBEDDING_API_TIMEOUT": 60.0,
    # Benchmark
    "BENCHMARK": "locomo",
    # LongHealthMem default: one text = one turn, no splitting unless explicitly enabled.
    "LONGHEALTHMEM_CHUNK_SIZE": 0,
    "LONGHEALTHMEM_CHUNK_OVERLAP": 0,
    "LONGHEALTHMEM_USER_SPEAKER": "patient",
    # Memory building
    "WINDOW_SIZE": 40,
    "OVERLAP_SIZE": 2,
    # Retrieval
    "SEMANTIC_TOP_K": 25,
    "KEYWORD_TOP_K": 5,
    "STRUCTURED_TOP_K": 5,
    # Storage
    "LANCEDB_PATH": "./lancedb_data",
    "MEMORY_TABLE_NAME": "memory_entries",
    # Parallel
    "ENABLE_PARALLEL_PROCESSING": True,
    "MAX_PARALLEL_WORKERS": 16,
    "ENABLE_PARALLEL_RETRIEVAL": True,
    "MAX_RETRIEVAL_WORKERS": 8,
    "ENABLE_PLANNING": True,
    "ENABLE_REFLECTION": True,
    "MAX_REFLECTION_ROUNDS": 2,
    # Judge (LoCoMo)
    "JUDGE_API_KEY": None,
    "JUDGE_BASE_URL": None,
    "JUDGE_MODEL": None,
    "JUDGE_ENABLE_THINKING": False,
    "JUDGE_USE_STREAMING": False,
    "JUDGE_TEMPERATURE": 0.3,
}


_ENV_ALIASES: Dict[str, Iterable[str]] = {
    "OPENAI_API_KEY": ("OPENAI_API_KEY",),
    "OPENAI_BASE_URL": ("OPENAI_BASE_URL",),
    "LLM_MODEL": ("LLM_MODEL",),
    "ENABLE_THINKING": ("ENABLE_THINKING",),
    "USE_STREAMING": ("USE_STREAMING",),
    "USE_JSON_FORMAT": ("USE_JSON_FORMAT",),
    "EMBEDDING_BACKEND": ("EMBEDDING_BACKEND", "EMBEDDING_PROVIDER"),
    "EMBEDDING_PROVIDER": ("EMBEDDING_PROVIDER",),
    "EMBEDDING_MODEL": ("EMBEDDING_MODEL",),
    "EMBEDDING_DIMENSION": ("EMBEDDING_DIMENSION",),
    "EMBEDDING_API_KEY": ("EMBEDDING_API_KEY", "DASHSCOPE_API_KEY"),
    "EMBEDDING_API_BASE": ("EMBEDDING_API_BASE",),
    "EMBEDDING_API_BATCH_SIZE": ("EMBEDDING_API_BATCH_SIZE",),
    "EMBEDDING_API_TIMEOUT": ("EMBEDDING_API_TIMEOUT",),
    "BENCHMARK": ("BENCHMARK",),
    "LONGHEALTHMEM_CHUNK_SIZE": ("LONGHEALTHMEM_CHUNK_SIZE",),
    "LONGHEALTHMEM_CHUNK_OVERLAP": ("LONGHEALTHMEM_CHUNK_OVERLAP",),
    "LONGHEALTHMEM_USER_SPEAKER": ("LONGHEALTHMEM_USER_SPEAKER",),
    "WINDOW_SIZE": ("WINDOW_SIZE",),
    "OVERLAP_SIZE": ("OVERLAP_SIZE",),
    "SEMANTIC_TOP_K": ("SEMANTIC_TOP_K",),
    "KEYWORD_TOP_K": ("KEYWORD_TOP_K",),
    "STRUCTURED_TOP_K": ("STRUCTURED_TOP_K",),
    "LANCEDB_PATH": ("LANCEDB_PATH",),
    "MEMORY_TABLE_NAME": ("MEMORY_TABLE_NAME",),
    "ENABLE_PARALLEL_PROCESSING": ("ENABLE_PARALLEL_PROCESSING",),
    "MAX_PARALLEL_WORKERS": ("MAX_PARALLEL_WORKERS",),
    "ENABLE_PARALLEL_RETRIEVAL": ("ENABLE_PARALLEL_RETRIEVAL",),
    "MAX_RETRIEVAL_WORKERS": ("MAX_RETRIEVAL_WORKERS",),
    "ENABLE_PLANNING": ("ENABLE_PLANNING",),
    "ENABLE_REFLECTION": ("ENABLE_REFLECTION",),
    "MAX_REFLECTION_ROUNDS": ("MAX_REFLECTION_ROUNDS",),
    "JUDGE_API_KEY": ("JUDGE_API_KEY",),
    "JUDGE_BASE_URL": ("JUDGE_BASE_URL",),
    "JUDGE_MODEL": ("JUDGE_MODEL",),
    "JUDGE_ENABLE_THINKING": ("JUDGE_ENABLE_THINKING",),
    "JUDGE_USE_STREAMING": ("JUDGE_USE_STREAMING",),
    "JUDGE_TEMPERATURE": ("JUDGE_TEMPERATURE",),
}


_BOOL_KEYS = {
    "ENABLE_THINKING",
    "USE_STREAMING",
    "USE_JSON_FORMAT",
    "ENABLE_PARALLEL_PROCESSING",
    "ENABLE_PARALLEL_RETRIEVAL",
    "ENABLE_PLANNING",
    "ENABLE_REFLECTION",
    "JUDGE_ENABLE_THINKING",
    "JUDGE_USE_STREAMING",
}

_INT_KEYS = {
    "EMBEDDING_DIMENSION",
    "EMBEDDING_API_BATCH_SIZE",
    "LONGHEALTHMEM_CHUNK_SIZE",
    "LONGHEALTHMEM_CHUNK_OVERLAP",
    "WINDOW_SIZE",
    "OVERLAP_SIZE",
    "SEMANTIC_TOP_K",
    "KEYWORD_TOP_K",
    "STRUCTURED_TOP_K",
    "MAX_PARALLEL_WORKERS",
    "MAX_RETRIEVAL_WORKERS",
    "MAX_REFLECTION_ROUNDS",
}

_FLOAT_KEYS = {
    "EMBEDDING_API_TIMEOUT",
    "JUDGE_TEMPERATURE",
}


def _to_bool(raw_value: str) -> bool:
    return str(raw_value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_user_config_values() -> Dict[str, Any]:
    if not _USER_CONFIG_PATH.exists():
        return {}

    try:
        spec = importlib.util.spec_from_file_location(
            "simplemem_user_config",
            _USER_CONFIG_PATH,
        )
        if spec is None or spec.loader is None:
            return {}

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {
            name: getattr(module, name)
            for name in dir(module)
            if name.isupper()
        }
    except Exception as exc:
        print(f"Warning: failed to load {_USER_CONFIG_PATH}: {exc}")
        return {}


def _coerce_env_value(key: str, raw_value: Optional[str]) -> Any:
    if raw_value is None:
        return None

    value = str(raw_value).strip()
    if value == "":
        return None

    if key in _BOOL_KEYS:
        return _to_bool(value)
    if key in _INT_KEYS:
        try:
            return int(value)
        except ValueError:
            return None
    if key in _FLOAT_KEYS:
        try:
            return float(value)
        except ValueError:
            return None
    return value


def _resolve_value(key: str, merged: Dict[str, Any]) -> Any:
    env_names = _ENV_ALIASES.get(key, ())
    for env_name in env_names:
        env_raw = os.getenv(env_name)
        if env_raw is None:
            continue
        env_value = _coerce_env_value(key, env_raw)
        if env_value is not None:
            return env_value
    return merged.get(key)


def _build_runtime_config() -> SimpleNamespace:
    merged: Dict[str, Any] = dict(_DEFAULTS)
    merged.update(_load_user_config_values())

    resolved: Dict[str, Any] = {}
    for key in merged:
        resolved[key] = _resolve_value(key, merged)

    # Expose whether config.py was found, useful for diagnostics.
    resolved["HAS_USER_CONFIG"] = _USER_CONFIG_PATH.exists()
    resolved["USER_CONFIG_PATH"] = str(_USER_CONFIG_PATH)
    return SimpleNamespace(**resolved)


config = _build_runtime_config()

