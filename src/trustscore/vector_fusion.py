from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .config import get_settings

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
except ImportError:  # pragma: no cover - optional heavy dep
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore


def _split_sentences(text: str, *, max_sentences: int = 4) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[\.!\?]\s+", text)
    sentences = [p.strip() for p in parts if len(p.strip()) > 15]
    return sentences[:max_sentences]


def _article_references(articles: Iterable[Dict[str, Any]], *, max_refs: int = 12) -> list[str]:
    refs: list[str] = []
    for item in articles:
        title = item.get("title") or ""
        snippet = item.get("snippet") or ""
        content = item.get("article") or ""
        combined = " ".join([title, snippet, content]).strip()
        if combined:
            refs.append(combined)
        if len(refs) >= max_refs:
            break
    return refs


class EmbeddingBackbone:
    """Shared embedding backbone using LaBSE (or configured) with safe fallback."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.semantic_model_name
        self._model = None

    def _ensure_model(self):
        if self._model is not None or SentenceTransformer is None:
            return
        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception:
            self._model = None

    def best_similarity(self, claims: list[str], references: list[str]) -> float:
        """Return best cosine similarity in [0, 1]."""
        if not claims or not references:
            return 0.0
        self._ensure_model()
        if self._model is None or util is None:
            return 0.0
        try:
            claim_emb = self._model.encode(claims, convert_to_tensor=True, normalize_embeddings=True)
            ref_emb = self._model.encode(references, convert_to_tensor=True, normalize_embeddings=True)
            scores = util.cos_sim(claim_emb, ref_emb)
            best = float(scores.max().item())
            # Normalize from [-1, 1] to [0, 1]
            return max(0.0, min(1.0, (best + 1) / 2))
        except Exception:
            return 0.0


@dataclass
class FusionWeights:
    linguistic: float = 0.2
    semantic: float = 0.35
    source: float = 0.25
    media: float = 0.1
    anomaly: float = 0.1

    def normalized(self) -> Dict[str, float]:
        total = self.linguistic + self.semantic + self.source + self.media + self.anomaly
        if total <= 0:
            return {k: 0.0 for k in ["linguistic", "semantic", "source", "media", "anomaly"]}
        return {
            "linguistic": self.linguistic / total,
            "semantic": self.semantic / total,
            "source": self.source / total,
            "media": self.media / total,
            "anomaly": self.anomaly / total,
        }


class MultiVectorFusion:
    """Compute multi-vector fusion for trust scoring."""

    def __init__(self, *, embedding_backbone: Optional[EmbeddingBackbone] = None, weights: Optional[FusionWeights] = None):
        self.embedding = embedding_backbone or EmbeddingBackbone()
        self.weights = weights or FusionWeights()
        self._weight_map = self.weights.normalized()

    def compute(
        self,
        original_text: str,
        found_articles: list[Dict[str, Any]],
        detectors: list[Any],
        domain_whitelist: list[str],
        domain_blacklist: list[str],
        llm_vectors: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        det_map = {d.name: d for d in detectors}
        llm_vectors = llm_vectors or {}

        # Semantic vector
        claims = _split_sentences(original_text)
        references = _article_references(found_articles)
        best_similarity = self.embedding.best_similarity(claims, references)
        semantic_component = self._semantic_component(best_similarity)

        # Linguistic vector derived from clickbait/spam detectors
        linguistic_component = self._linguistic_component(det_map)

        # Source vector from authority + blacklist overlap
        source_component = self._source_component(det_map, found_articles, domain_whitelist, domain_blacklist)

        # Media vector from media-age detector
        media_component = self._media_component(det_map)

        # Anomaly vector from spam-behavior + domain similarity
        anomaly_component = self._anomaly_component(det_map)

        fused_score = (
            semantic_component * self._weight_map["semantic"]
            + linguistic_component * self._weight_map["linguistic"]
            + source_component * self._weight_map["source"]
            + media_component * self._weight_map["media"]
            + anomaly_component * self._weight_map["anomaly"]
        )

        return {
            "semantic_similarity": round(best_similarity, 3),
            "components": {
                "semantic": round(semantic_component, 3),
                "linguistic": round(linguistic_component, 3),
                "source": round(source_component, 3),
                "media": round(media_component, 3),
                "anomaly": round(anomaly_component, 3),
            },
            "fused_score": round(fused_score, 3),
            "llm_vectors": {
                "V_semantic_llm": round(llm_vectors.get("V_semantic_llm", 0.0), 3),
                "V_linguistic_llm": round(llm_vectors.get("V_linguistic_llm", 0.0), 3),
                "V_intent_llm": round(llm_vectors.get("V_intent_llm", 0.0), 3),
            },
        }

    @staticmethod
    def _semantic_component(similarity: float) -> float:
        # High similarity strongly boosts; very low similarity penalizes.
        if similarity >= 0.85:
            return 1.0
        if similarity <= 0.30:
            return 0.15
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def _linguistic_component(detectors: Dict[str, Any]) -> float:
        base = 1.0
        clickbait_penalty = abs(detectors.get("clickbait", object()).score_delta) / 50 if "clickbait" in detectors else 0.0
        spam_penalty = abs(detectors.get("spam-behavior", object()).score_delta) / 60 if "spam-behavior" in detectors else 0.0
        ai_penalty = 0.25 if "ai-generated" in (detectors.get("ai-content", object()).flags if "ai-content" in detectors else []) else 0.0
        penalty = min(0.6, clickbait_penalty + spam_penalty + ai_penalty)
        return max(0.0, base - penalty)

    @staticmethod
    def _source_component(detectors: Dict[str, Any], found_articles: list[Dict[str, Any]], whitelist: list[str], blacklist: list[str]) -> float:
        authority = detectors.get("authority")
        trusted_domains = set(authority.meta.get("trusted_domains", [])) if authority and authority.meta else set()
        whitelist_hits = min(1.0, len(trusted_domains) / 3) if trusted_domains else 0.0

        domains = {str(a.get("domain") or "").lower() for a in found_articles if a.get("domain")}
        blacklist_hits = {d for d in domains if d in {b.lower() for b in blacklist}}
        blacklist_penalty = 0.4 if blacklist_hits else 0.0

        gov_bonus = 0.1 if any(d.endswith(".gov.vn") or d.endswith(".gov") for d in trusted_domains) else 0.0
        base = 0.3 + whitelist_hits + gov_bonus
        return max(0.0, min(1.0, base - blacklist_penalty))

    @staticmethod
    def _media_component(detectors: Dict[str, Any]) -> float:
        media_det = detectors.get("media-age")
        if media_det and "old-media" in media_det.flags:
            return 0.35
        return 0.85

    @staticmethod
    def _anomaly_component(detectors: Dict[str, Any]) -> float:
        spam_det = detectors.get("spam-behavior")
        domain_det = detectors.get("domain-similarity")
        penalty = 0.0
        if spam_det and spam_det.score_delta < -5:
            penalty += min(0.4, abs(spam_det.score_delta) / 40)
        if domain_det and "typosquatting" in domain_det.flags:
            penalty += 0.2
        return max(0.0, 0.8 - penalty)
