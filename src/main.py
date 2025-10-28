from fastapi import FastAPI

from trustscore.analyzers import ImageVerifier, LanguageRiskScorer, SemanticAnalyzer
from trustscore.models import ClaimPayload, TrustScoreResult
from trustscore.reputation import ReputationClient
from trustscore.sources import SourceAggregator
from trustscore.trust_engine import TrustEngine

app = FastAPI(title="TrustScore Verification Service", version="0.1.0")
engine = TrustEngine(
    aggregator=SourceAggregator(),
    reputation_client=ReputationClient(),
    semantic_analyzer=SemanticAnalyzer(),
    language_scorer=LanguageRiskScorer(),
    image_verifier=ImageVerifier(),
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/verify", response_model=TrustScoreResult)
async def verify(payload: ClaimPayload) -> TrustScoreResult:
    return await engine.verify_claim(payload)
