from __future__ import annotations

from dataclasses import dataclass, field

from .analyzers import ImageVerifier, LanguageRiskScorer, SemanticAnalyzer
from .config import get_settings
from .models import (
    AlternativeLink,
    ClaimPayload,
    ReputationProfile,
    SourceEvidence,
    TrustScoreResult,
)
from .reputation import ReputationClient
from .sources import SourceAggregator


@dataclass
class TrustWeights:
    sources: float = 0.4
    semantic: float = 0.3
    image: float = 0.2
    language: float = 0.1

    def as_dict(self) -> dict[str, float]:
        total = self.sources + self.semantic + self.image + self.language
        if total <= 0:
            raise ValueError("TrustWeights sum must be positive")
        return {
            "sources": self.sources / total,
            "semantic": self.semantic / total,
            "image": self.image / total,
            "language": self.language / total,
        }


@dataclass
class TrustEngine:
    aggregator: SourceAggregator
    reputation_client: ReputationClient
    semantic_analyzer: SemanticAnalyzer
    language_scorer: LanguageRiskScorer
    image_verifier: ImageVerifier
    weights: TrustWeights = field(default_factory=TrustWeights)
    threshold: float = 0.85

    def __post_init__(self) -> None:
        self._settings = get_settings()
        self._weight_map = self.weights.as_dict()
        self._trusted_domains = {domain.lower() for domain in self._settings.trusted_domains}
        self._suspicious_domains = {domain.lower() for domain in self._settings.suspicious_domains}

    async def verify_claim(self, payload: ClaimPayload) -> TrustScoreResult:
        primary_document = payload.primary_document
        claim_text = payload.claim_text
        claim_url = (
            str(payload.url)
            if payload.url
            else (str(primary_document.url) if primary_document and primary_document.url else None)
        )
        evidence = await self.aggregator.gather(
            claim_text,
            claim_url,
            provided_documents=payload.supporting_documents,
        )
        if not evidence:
            return TrustScoreResult(
                trust_score=0.2,
                verdict="insufficient-evidence",
                evidence=[],
                alternatives=[],
                debug={"message": "No trusted sources matched the claim."},
            )

        reputations = await self.reputation_client.batch_fetch([item.source for item in evidence])
        enriched = [
            self._adjust_with_reputation(item, reputations.get(item.source))
            for item in evidence
        ]

        source_score = self._aggregate_source_signal(enriched)
        context_text = payload.context or claim_text
        semantic_score = self._normalize_similarity(
            await self.semantic_analyzer.score(
                context_text,
                [item.content or item.excerpt or "" for item in enriched],
            )
        )
        language_score = self.language_scorer.score(
            context_text,
            language=payload.language,
        )
        image_score = await self.image_verifier.score(payload.media_hashes)

        components = {
            "sources": source_score,
            "semantic": semantic_score,
            "image": image_score,
            "language": language_score,
        }
        trust_score = self._combine_components(components)
        verdict = self._derive_verdict(trust_score)
        alternatives = self._build_alternatives(enriched, limit=3)
        return TrustScoreResult(
            trust_score=trust_score,
            verdict=verdict,
            evidence=enriched,
            alternatives=alternatives if trust_score < self.threshold else [],
            components=components,
            debug={
                "sources": list({item.source for item in enriched}),
                "weights": self._weight_map,
            },
        )

    def _adjust_with_reputation(
        self,
        evidence: SourceEvidence,
        profile: ReputationProfile | None,
    ) -> SourceEvidence:
        base = evidence.raw_score
        domain_modifier = self._domain_modifier(evidence.source)
        combined_score = base * domain_modifier
        if profile:
            combined_score = (combined_score * 0.6) + (profile.score * 0.4)
            if profile.verified:
                combined_score += 0.05
        adjusted = min(1.0, combined_score)
        return evidence.model_copy(update={"confidence": adjusted})

    def _aggregate_source_signal(self, evidence: list[SourceEvidence]) -> float:
        if not evidence:
            return 0.0
        total_confidence = sum(item.confidence for item in evidence)
        top_confidence = max(item.confidence for item in evidence)
        normalized = (total_confidence / len(evidence) + top_confidence) / 2
        return round(min(1.0, normalized), 3)

    def _combine_components(self, components: dict[str, float]) -> float:
        total = 0.0
        for key, weight in self._weight_map.items():
            total += components.get(key, 0.0) * weight
        return round(min(1.0, max(0.0, total)), 3)

    def _derive_verdict(self, score: float) -> str:
        if score >= self.threshold:
            return "verified"
        if score >= 0.5:
            return "needs-review"
        return "likely-false"

    def _domain_modifier(self, domain: str) -> float:
        domain_lower = domain.lower()
        if domain_lower in self._trusted_domains:
            return 1.1
        if domain_lower in self._suspicious_domains:
            return 0.6
        return 1.0

    @staticmethod
    def _normalize_similarity(value: float) -> float:
        # sentence-transformers similarity can be [-1, 1]; convert to [0, 1]
        normalized = (value + 1) / 2
        return round(min(1.0, max(0.0, normalized)), 3)

    def _build_alternatives(self, evidence: list[SourceEvidence], limit: int) -> list[AlternativeLink]:
        sorted_evidence = sorted(evidence, key=lambda item: item.confidence, reverse=True)
        alternatives = []
        for item in sorted_evidence:
            alternatives.append(AlternativeLink(source=item.source, url=item.url, title=item.title))
            if len(alternatives) >= limit:
                break
        return alternatives
