"""
Embedding utilities with pluggable providers.

Supports:
- Local SentenceTransformers models (existing behavior)
- API-based OpenAI-compatible embedding endpoints
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from utils.config_loader import config


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _resolve_embedding_api_key() -> Optional[str]:
    return (
        _safe_str(os.getenv("EMBEDDING_API_KEY"))
        or _safe_str(os.getenv("DASHSCOPE_API_KEY"))
        or _safe_str(getattr(config, "EMBEDDING_API_KEY", None))
        or _safe_str(getattr(config, "DASHSCOPE_API_KEY", None))
        or _safe_str(os.getenv("OPENAI_API_KEY"))
        or _safe_str(getattr(config, "OPENAI_API_KEY", None))
    )


def _resolve_embedding_api_base() -> Optional[str]:
    return (
        _safe_str(os.getenv("EMBEDDING_API_BASE"))
        or _safe_str(getattr(config, "EMBEDDING_API_BASE", None))
        or _safe_str(os.getenv("OPENAI_BASE_URL"))
        or _safe_str(getattr(config, "OPENAI_BASE_URL", None))
    )


def _is_dashscope_endpoint(api_base: Optional[str]) -> bool:
    return bool(api_base and "dashscope.aliyuncs.com" in api_base.lower())


def _is_dashscope_text_embedding_v4(model_name: str, api_base: Optional[str]) -> bool:
    if not model_name:
        return False
    return _is_dashscope_endpoint(api_base) and model_name.strip().lower().startswith("text-embedding-v4")


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


class BaseEmbeddingProvider(ABC):
    model_name: str
    model_type: str
    dimension: int
    supports_query_prompt: bool = False

    @abstractmethod
    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError


class LocalSentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Local embedding provider based on SentenceTransformers.
    """

    def __init__(self, model_name: str, use_optimization: bool = True):
        self.model_name = model_name
        self.use_optimization = use_optimization

        print(f"Loading local embedding model: {self.model_name}")

        if self.model_name.startswith("qwen3"):
            self._init_qwen3_sentence_transformer()
        else:
            self._init_standard_sentence_transformer()

    def _init_qwen3_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer

            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
            }

            model_path = qwen3_models.get(self.model_name, self.model_name)
            print(f"Loading Qwen3 model via SentenceTransformers: {model_path}")

            if self.use_optimization:
                try:
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2",
                            "device_map": "auto",
                        },
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True,
                    )
                    print("Qwen3 loaded with flash_attention_2 optimization")
                except Exception as exc:
                    print(f"Flash attention optimization failed ({exc}), using standard loading")
                    self.model = SentenceTransformer(model_path, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_path, trust_remote_code=True)

            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "qwen3_sentence_transformer"
            self.supports_query_prompt = hasattr(self.model, "prompts") and "query" in getattr(self.model, "prompts", {})

            print(f"Qwen3 model loaded successfully with dimension: {self.dimension}")
            if self.supports_query_prompt:
                print("Query prompt support detected")

        except Exception as exc:
            print(f"Failed to load Qwen3 model: {exc}")
            print("Falling back to default SentenceTransformers model")
            self._fallback_to_sentence_transformer()

    def _init_standard_sentence_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "sentence_transformer"
            self.supports_query_prompt = False
            print(f"SentenceTransformer model loaded with dimension: {self.dimension}")
        except Exception as exc:
            print(f"Failed to load SentenceTransformer model: {exc}")
            raise

    def _fallback_to_sentence_transformer(self):
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Using fallback model: {fallback_model}")
        self.model_name = fallback_model
        self._init_standard_sentence_transformer()

    def _encode_with_query_prompt(self, texts: List[str]) -> np.ndarray:
        try:
            embeddings = self.model.encode(
                texts,
                prompt_name="query",
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            print(f"Query prompt encoding failed: {exc}, falling back to standard encoding")
            return self._encode_standard(texts)

    def _encode_standard(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if self.model_type == "qwen3_sentence_transformer" and self.supports_query_prompt and is_query:
            return self._encode_with_query_prompt(texts)
        return self._encode_standard(texts)


class OpenAICompatibleEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider for OpenAI-compatible API endpoints.

    Environment-oriented settings:
    - EMBEDDING_API_KEY
    - EMBEDDING_API_BASE
    - EMBEDDING_MODEL
    - EMBEDDING_DIMENSION

    Notes:
    - Query-specific SDK features (for example DashScope native `text_type="query"`
      or `instruct`) are not available in this OpenAI-compatible path.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base: Optional[str],
        dimension: int,
        batch_size: int = 64,
        timeout: Optional[float] = None,
    ):
        if not api_key:
            raise ValueError(
                "Missing embedding API key. Set EMBEDDING_API_KEY (or DASHSCOPE_API_KEY)."
            )
        if not model_name:
            raise ValueError("Missing embedding model. Set EMBEDDING_MODEL.")
        if not dimension or int(dimension) <= 0:
            raise ValueError(
                "Missing/invalid embedding dimension. Set EMBEDDING_DIMENSION to a positive integer."
            )

        from openai import OpenAI

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name
        self.model_type = "openai_compatible_api"
        self.dimension = int(dimension)

        requested_batch_size = max(1, int(batch_size))
        if _is_dashscope_text_embedding_v4(self.model_name, api_base) and requested_batch_size > 10:
            print(
                "Capping embedding batch size to 10 for DashScope text-embedding-v4 compatibility."
            )
            requested_batch_size = 10
        self.batch_size = requested_batch_size
        self.supports_query_prompt = False

        endpoint_msg = api_base if api_base else "default OpenAI endpoint"
        print(
            f"Using API embedding model: {self.model_name} "
            f"(dim={self.dimension}, batch={self.batch_size}, endpoint={endpoint_msg})"
        )

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        del is_query  # Interface compatibility only. No query-only API parameters are injected.

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors: List[List[float]] = []
        for batch in _batched(texts, self.batch_size):
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=self.dimension,
            )
            vectors.extend(item.embedding for item in response.data)

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {matrix.shape[1]} from model '{self.model_name}'."
            )

        return _l2_normalize(matrix)


class EmbeddingModel:
    """
    Unified embedding facade with local and API providers.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_optimization: bool = True,
        provider: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        backend = (
            provider
            or os.getenv("EMBEDDING_BACKEND")
            or os.getenv("EMBEDDING_PROVIDER")
            or getattr(config, "EMBEDDING_BACKEND", None)
            or getattr(config, "EMBEDDING_PROVIDER", "local")
        )
        backend = str(backend).strip().lower()

        resolved_model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL")
            or getattr(config, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )

        resolved_dimension = (
            dimension
            or _safe_int(os.getenv("EMBEDDING_DIMENSION"))
            or _safe_int(getattr(config, "EMBEDDING_DIMENSION", None))
        )

        if backend in {"api", "openai", "openai_compatible"}:
            api_key = _resolve_embedding_api_key()
            api_base = _resolve_embedding_api_base()
            api_batch_size = (
                _safe_int(os.getenv("EMBEDDING_API_BATCH_SIZE"))
                or _safe_int(getattr(config, "EMBEDDING_API_BATCH_SIZE", None))
                or 64
            )
            api_timeout = (
                _safe_float(os.getenv("EMBEDDING_API_TIMEOUT"))
                or _safe_float(getattr(config, "EMBEDDING_API_TIMEOUT", None))
            )

            if not api_key:
                raise ValueError(
                    "Embedding backend is 'api' but no API key was found. "
                    "Set EMBEDDING_API_KEY (or DASHSCOPE_API_KEY)."
                )
            if not resolved_model_name:
                raise ValueError(
                    "Embedding backend is 'api' but EMBEDDING_MODEL is missing."
                )
            if not resolved_dimension or int(resolved_dimension) <= 0:
                raise ValueError(
                    "Embedding backend is 'api' but EMBEDDING_DIMENSION is missing or invalid."
                )

            self._provider = OpenAICompatibleEmbeddingProvider(
                model_name=resolved_model_name,
                api_key=api_key,
                api_base=api_base,
                dimension=int(resolved_dimension),
                batch_size=api_batch_size,
                timeout=api_timeout,
            )
        else:
            self._provider = LocalSentenceTransformerProvider(
                model_name=resolved_model_name,
                use_optimization=use_optimization,
            )

        self.model_name = self._provider.model_name
        self.model_type = self._provider.model_type
        self.dimension = self._provider.dimension
        self.supports_query_prompt = getattr(self._provider, "supports_query_prompt", False)

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self._provider.encode(texts, is_query=is_query)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]

    def encode_query(self, queries: List[str]) -> np.ndarray:
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        return self.encode(documents, is_query=False)
