from __future__ import annotations

import asyncio
import re
import numpy as np
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TYPE_CHECKING, Optional

import httpx

from .config import get_settings
from .llm_adapter import LLMAdapter, LLMFallbackError

try:  # pragma: no cover - optional heavy dependency
    from sentence_transformers import SentenceTransformer, util
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from sentence_transformers import SentenceTransformer as SentenceTransformerType
else:  # pragma: no cover
    SentenceTransformerType = Any


class SemanticAnalyzer:
    """Compare a claim against reference texts using sentence embeddings."""

    def __init__(self, *, model_name: str | None = None, max_workers: int = 1, llm: Optional[LLMAdapter] = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.semantic_model_name
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._model: SentenceTransformerType | None = None
        self._llm = llm
        self.last_llm_similarity: float | None = None

    def _ensure_model(self) -> SentenceTransformerType | None:
        if self._model is None and SentenceTransformer is not None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def score(self, claim_text: str, references: Sequence[str]) -> float:
        texts = [text for text in references if text]
        if not claim_text or not texts:
            return 0.5
        base_score = 0.5
        model = self._ensure_model()
        if model is not None and util is not None:
            loop = asyncio.get_running_loop()
            base_score = await loop.run_in_executor(
                self._executor,
                self._compute_similarity,
                model,
                claim_text,
                texts,
            )

        llm_score = self._llm_similarity(claim_text, texts)
        if llm_score is not None:
            self.last_llm_similarity = llm_score
            return round(max(base_score, llm_score), 3)
        return round(base_score, 3)

    @staticmethod
    def _compute_similarity(
        model: SentenceTransformerType,
        claim_text: str,
        texts: Sequence[str],
    ) -> float:
        embeddings = model.encode([claim_text, *texts], convert_to_tensor=True, normalize_embeddings=True)
        claim_embedding = embeddings[0]
        reference_embeddings = embeddings[1:]
        similarities = util.dot_score(claim_embedding, reference_embeddings).tolist()[0]
        best_match = max(similarities, default=0.0)
        return round(float(best_match), 3)

    def _llm_similarity(self, claim_text: str, references: Sequence[str]) -> float | None:
        if not self._llm or not references:
            return None
        try:
            claim_vec = self._llm.embedding(claim_text)
            ref_vecs = [self._llm.embedding(ref) for ref in references[:8]]
        except LLMFallbackError:
            return None
        claim_arr = self._to_array(claim_vec)
        best = 0.0
        for vec in ref_vecs:
            ref_arr = self._to_array(vec)
            denom = (np.linalg.norm(claim_arr) * np.linalg.norm(ref_arr)) or 1e-6
            score = float(np.dot(claim_arr, ref_arr) / denom)
            best = max(best, score)
        # normalize [-1,1] to [0,1]
        return max(0.0, min(1.0, (best + 1) / 2))

    @staticmethod
    def _to_array(vec: list[float]) -> np.ndarray:
        return np.array(vec, dtype=float)


class LanguageRiskScorer:
    CLICKBAIT_PATTERNS = (
        r"!{2,}",
        r"\b(?:giật gân|sốc|chấn động|bạn sẽ không tin|không ai ngờ)\b",
        r"\b(?:100%|cam kết|đảm bảo|chưa từng có)\b",
    )
    ALL_CAPS_PATTERN = re.compile(r"[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]{5,}")

    def __init__(self, llm: Optional[LLMAdapter] = None) -> None:
        self._llm = llm
        self.last_llm_label: str | None = None
        self.last_llm_score: float | None = None

    def score(self, text: str, *, language: str | None = None) -> float:
        if not text:
            return 0.5
        penalties = 0.0
        word_count = max(len(text.split()), 1)
        exclamation_ratio = text.count("!") / word_count
        if exclamation_ratio > 0.05:
            penalties += 0.1
        if self.ALL_CAPS_PATTERN.search(text):
            penalties += 0.15
        for pattern in self.CLICKBAIT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                penalties += 0.1
        if language and language.lower().startswith("vi"):
            penalties *= 1.1

        credible_score = max(0.0, 1.0 - min(0.6, penalties))

        llm_score = self._llm_classification_score(text)
        if llm_score is not None:
            combined = (credible_score + llm_score) / 2
            return round(combined, 3)

        return round(credible_score, 3)

    def _llm_classification_score(self, text: str) -> float | None:
        if not self._llm:
            return None
        labels = ["clickbait", "hedging", "rumor", "ai-generated", "neutral"]
        try:
            label = self._llm.classify(text, labels)
        except LLMFallbackError:
            return None
        self.last_llm_label = label
        mapping = {
            "clickbait": 0.45,
            "hedging": 0.55,
            "rumor": 0.5,
            "ai-generated": 0.5,
            "neutral": 0.9,
        }
        self.last_llm_score = mapping.get(label, 0.6)
        return self.last_llm_score


class ImageVerifier:
    """Proxy to reverse-image search providers. Returns confidence 0-1."""

    def __init__(self, *, timeout: float = 12.0) -> None:
        settings = get_settings()
        self._timeout = timeout
        self._endpoint = settings.image_search_api_url
        self._api_key = settings.image_search_api_key

    async def score(self, media_hashes: Sequence[str]) -> float:
        if not media_hashes:
            return 0.5
        if not self._endpoint or not self._api_key:
            return 0.45
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [self._query_hash(client, media_hash) for media_hash in media_hashes[:5]]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        scores = [response for response in responses if isinstance(response, float)]
        if not scores:
            return 0.45
        return round(sum(scores) / len(scores), 3)

    async def _query_hash(self, client: httpx.AsyncClient, media_hash: str) -> float:
        payload = {"hash": media_hash}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            response = await client.post(self._endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return float(data.get("confidence", 0.5))
        except httpx.HTTPError:
            return 0.45
