from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class DocumentPayload(BaseModel):
    id: str = Field(..., min_length=1)
    role: Literal["primary", "support"] = "support"
    title: str | None = None
    url: HttpUrl | None = None
    source: str | None = None
    content: str | None = None
    excerpt: str | None = None
    published_at: datetime | None = None
    verified_origin: bool = False
    confidence_hint: float | None = Field(default=None, ge=0.0, le=1.0)
    language: str | None = None


class ClaimPayload(BaseModel):
    claim_text: str | None = Field(default=None, min_length=5)
    url: HttpUrl | None = None
    media_hashes: list[str] = Field(default_factory=list)
    language: str | None = None
    context: str | None = None
    documents: list[DocumentPayload] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ensure_claim_text(self) -> "ClaimPayload":
        if not self.claim_text:
            primary = self.primary_document
            if primary and primary.content:
                self.claim_text = primary.content
            if not self.url and primary and primary.url:
                self.url = primary.url
            if not self.language and primary and primary.language:
                self.language = primary.language
        if not self.claim_text or len(self.claim_text.strip()) < 5:
            raise ValueError("Either claim_text or a primary document with at least 5 characters is required.")
        return self

    @property
    def primary_document(self) -> DocumentPayload | None:
        for document in self.documents:
            if document.role == "primary":
                return document
        return None

    @property
    def supporting_documents(self) -> list[DocumentPayload]:
        return [document for document in self.documents if document.role != "primary"]


class SourceEvidence(BaseModel):
    source: str
    url: HttpUrl
    title: str
    excerpt: str | None = None
    content: str | None = None
    published_at: datetime | None = None
    verified_origin: bool = False
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    raw_score: float = 0.0


class ReputationProfile(BaseModel):
    source: str
    score: float = Field(..., ge=0.0, le=1.0)
    verified: bool = False
    last_audited: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlternativeLink(BaseModel):
    source: str
    url: HttpUrl
    title: str


class TrustScoreResult(BaseModel):
    trust_score: float = Field(..., ge=0.0, le=1.0)
    verdict: str
    evidence: list[SourceEvidence]
    alternatives: list[AlternativeLink] = Field(default_factory=list)
    components: dict[str, float] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)
