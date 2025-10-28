from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TYPE_CHECKING

import httpx

from .config import get_settings

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

    def __init__(self, *, model_name: str | None = None, max_workers: int = 1) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.semantic_model_name
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._model: SentenceTransformerType | None = None

    def _ensure_model(self) -> SentenceTransformerType | None:
        if self._model is None and SentenceTransformer is not None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def score(self, claim_text: str, references: Sequence[str]) -> float:
        texts = [text for text in references if text]
        if not claim_text or not texts:
            return 0.5
        model = self._ensure_model()
        if model is None or util is None:
            return 0.5
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            self._compute_similarity,
            model,
            claim_text,
            texts,
        )

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


class LanguageRiskScorer:
    CLICKBAIT_PATTERNS = (
        r"!{2,}",
        r"\b(?:giật gân|sốc|chấn động|bạn sẽ không tin|không ai ngờ)\b",
        r"\b(?:100%|cam kết|đảm bảo|chưa từng có)\b",
    )
    ALL_CAPS_PATTERN = re.compile(r"[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]{5,}")

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
        return round(credible_score, 3)


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
