"""
Offline LLM adapter (llama.cpp + sentence-transformers) with strict fallback.
Provides lightweight generate / embedding / classify helpers that must never
call external services. On any load/timeout error, raises LLMFallbackError so
callers can return to heuristic-only mode.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import get_settings

logger = logging.getLogger(__name__)


class LLMFallbackError(RuntimeError):
    """Raised when the offline LLM layer must be skipped."""


@dataclass
class _CacheEntry:
    value: Any
    hit: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


class LLMAdapter:
    """
    Offline LLM helper using llama.cpp + local embeddings.
    - generate: small deterministic completions
    - embedding: vector from sentence-transformers (or llama if supported)
    - classify: zero-shot via cosine similarity
    """

    def __init__(self, *, max_workers: int = 1) -> None:
        settings = get_settings()
        self._model_path = os.getenv("LLM_MODEL_PATH") or settings.llm_model_path
        self._embed_model_name = os.getenv("EMBED_MODEL_NAME") or settings.embed_model_name
        self._timeout = float(os.getenv("LLM_TIMEOUT", settings.llm_timeout))
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._llm = None
        self._embed_model = None
        self._cache: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self._llm_logs: List[Dict[str, Any]] = []

    # Public API -----------------------------------------------------
    def generate(self, prompt: str, timeout: float | None = None) -> str:
        cache_key = self._cache_key("gen", prompt)
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        llm = self._ensure_llm()
        deadline = timeout or self._timeout
        try:
            future = self._executor.submit(self._generate_sync, llm, prompt)
            output = future.result(timeout=deadline)
        except TimeoutError as exc:  # pragma: no cover - depends on runtime
            self._log_fallback("generate-timeout", prompt, str(exc))
            raise LLMFallbackError("LLM generate timeout") from exc
        except Exception as exc:  # pragma: no cover - runtime safety
            self._log_fallback("generate-error", prompt, str(exc))
            raise LLMFallbackError(f"LLM generate failed: {exc}") from exc
        self._set_cache(cache_key, output, prompt=prompt, output=output)
        return output

    def embedding(self, text: str) -> List[float]:
        cache_key = self._cache_key("emb", text)
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        model = self._ensure_embedder()
        try:
            vector = self._encode_embedding(model, text)
        except Exception as exc:  # pragma: no cover - runtime safety
            self._log_fallback("embedding-error", text, str(exc))
            raise LLMFallbackError(f"LLM embedding failed: {exc}") from exc
        self._set_cache(cache_key, vector, prompt=text, output_len=len(vector))
        return vector

    def classify(self, text: str, labels: List[str], timeout: float | None = None) -> str:
        """
        Zero-shot classification by embedding similarity.
        If labels are empty or embeddings fail -> fallback error.
        """
        if not labels:
            raise LLMFallbackError("No labels provided")

        try:
            text_vec = self.embedding(text)
            label_vecs = [(label, self.embedding(label)) for label in labels]
        except LLMFallbackError as exc:
            self._log_fallback("classify-embedding-missing", text, str(exc))
            raise

        text_arr = np.array(text_vec)
        best_label = labels[0]
        best_score = -1.0
        for label, vec in label_vecs:
            score = self._cosine(text_arr, np.array(vec))
            if score > best_score:
                best_label = label
                best_score = score

        self._set_cache(
            self._cache_key("cls", text + "::" + "|".join(labels)),
            best_label,
            prompt=text,
            output=best_label,
            labels=labels,
            score=best_score,
        )
        return best_label

    @property
    def logs(self) -> List[Dict[str, Any]]:
        return list(self._llm_logs)

    # Internal helpers ----------------------------------------------
    def _ensure_llm(self):
        if self._llm is not None:
            return self._llm
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as exc:
            raise LLMFallbackError(f"llama_cpp not available: {exc}") from exc
        if not self._model_path:
            raise LLMFallbackError("LLM_MODEL_PATH not configured")
        try:
            self._llm = Llama(model_path=self._model_path, n_threads=4)
            self._log_event("llm-load", {"model_path": self._model_path})
            return self._llm
        except Exception as exc:
            raise LLMFallbackError(f"Failed to load LLM: {exc}") from exc

    def _ensure_embedder(self):
        if self._embed_model is not None:
            return self._embed_model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise LLMFallbackError(f"sentence-transformers not available: {exc}") from exc
        try:
            self._embed_model = SentenceTransformer(self._embed_model_name)
            self._log_event("embedder-load", {"model": self._embed_model_name})
            return self._embed_model
        except Exception as exc:
            raise LLMFallbackError(f"Failed to load embedder: {exc}") from exc

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-6
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _generate_sync(llm, prompt: str) -> str:
        # llama_cpp returns {"choices": [{"text": "..."}]}
        response = llm(prompt, max_tokens=256, temperature=0.2, stop=["</s>", "###"])
        text = ""
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices and "text" in choices[0]:
                text = choices[0]["text"]
            elif isinstance(response.get("content"), str):
                text = response["content"]
        return (text or "").strip()

    def _encode_embedding(self, model, text: str) -> List[float]:
        vector = model.encode([text], normalize_embeddings=True)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if isinstance(vector, list) and vector and isinstance(vector[0], list):
            vector = vector[0]
        if not isinstance(vector, list):
            vector = list(vector)
        return [float(v) for v in vector]

    def _cache_key(self, prefix: str, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"{prefix}:{digest}"

    def _get_cache(self, key: str) -> Any | None:
        with self._lock:
            entry = self._cache.get(key)
        if entry:
            entry.hit = True
            return entry.value
        return None

    def _set_cache(self, key: str, value: Any, **meta: Any) -> None:
        with self._lock:
            self._cache[key] = _CacheEntry(value=value, hit=False, meta=meta)
        meta = {**meta, "key": key}
        self._log_event("cache-set", meta)

    def _log_event(self, event: str, payload: Dict[str, Any]) -> None:
        short_payload = payload.copy()
        if "prompt" in short_payload:
            short_payload["prompt"] = (short_payload["prompt"] or "")[:200]
        if "output" in short_payload and isinstance(short_payload["output"], str):
            short_payload["output"] = short_payload["output"][:200]
        short_payload["event"] = event
        self._llm_logs.append(short_payload)
        logger.debug("LLM event: %s", json.dumps(short_payload, ensure_ascii=False))

    def _log_fallback(self, reason: str, prompt: str, error: str) -> None:
        self._log_event(
            "fallback",
            {"reason": reason, "prompt": prompt[:200], "error": error[:200]},
        )

