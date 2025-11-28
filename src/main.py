"""
TrustScore Verification Service - Production Ready API
Main endpoint for fact-checking with AI processing
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict, Counter
import httpx
import asyncio
import uuid
import hashlib
import logging
import time
import os
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import tldextract
import unicodedata

from dotenv import load_dotenv
from trustscore.analyzers import ImageVerifier, LanguageRiskScorer, SemanticAnalyzer
from trustscore.llm_adapter import LLMAdapter, LLMFallbackError
from trustscore.models import ClaimPayload, TrustScoreResult
from trustscore.reputation import ReputationClient
from trustscore.sources import SourceAggregator
from trustscore.trust_engine import TrustEngine
from data_loader import load_datasets

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/trustscore.log') if os.path.exists('logs') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)
DATASETS: Dict[str, Any] = {}

# Load environment variables early so Config picks them up
load_dotenv()


# Service configuration
class Config:
    """Service configuration from environment variables"""
    
    VERSION = "2.0.0"
    TITLE = "TrustScore Verification Service"
    DESCRIPTION = "Production-ready fact-checking API with AI"
    
    CRAWLER_API_URL = os.getenv("CRAWLER_API_URL", "http://localhost:8000")
    CRAWLER_TIMEOUT = int(os.getenv("CRAWLER_TIMEOUT", "600"))
    
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_TTL_JOBS = int(os.getenv("REDIS_TTL_JOBS", "86400"))
    REDIS_TTL_CACHE = int(os.getenv("REDIS_TTL_CACHE", "3600"))
    
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
    MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "50"))
    JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "300"))
    DATA_DIR = os.getenv("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
    
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    @classmethod
    def validate(cls):
        """Log current configuration"""
        logger.info("Configuration loaded:")
        logger.info(f"  Crawler URL: {cls.CRAWLER_API_URL}")
        logger.info(f"  Redis URL: {cls.REDIS_URL}")
        logger.info(f"  Max concurrent jobs: {cls.MAX_CONCURRENT_JOBS}")
        logger.info(f"  Data dir: {cls.DATA_DIR}")


config = Config()


def extract_domain(url: Optional[str]) -> Optional[str]:
    """Extract normalized domain from a URL (including subdomains)."""
    if not url:
        return None
    try:
        parsed = tldextract.extract(url)
        if parsed.domain and parsed.suffix:
            return f"{parsed.domain}.{parsed.suffix}".lower()
    except Exception:
        pass
    try:
        hostname = urlparse(url if "://" in url else f"http://{url}").hostname
        return hostname.lower() if hostname else None
    except Exception:
        return None


def is_domain_whitelisted(domain: Optional[str], whitelist: List[str]) -> bool:
    """Return True if domain matches any whitelist entry."""
    if not domain:
        return False
    domain_lower = domain.lower()
    return any(domain_lower.endswith(w.lower()) or w.lower() in domain_lower for w in whitelist)


def _normalize_text(value: Optional[str]) -> str:
    """Normalize text for fuzzy comparison (lowercase, remove accents/spaces)."""
    if not value:
        return ""
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return "".join(ch for ch in value.lower() if ch.isalnum())


def match_trusted_page(author: Optional[str], trusted_sources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return trusted page entry if author matches a trusted page."""
    if not author:
        return None
    norm_author = _normalize_text(author)
    if not norm_author:
        return None
    for entry in trusted_sources:
        if entry.get("type") != "page":
            continue
        name = entry.get("name") or ""
        identifier = entry.get("identifier") or ""
        slug = identifier.split("/")[-1] if identifier else ""
        candidates = [
            _normalize_text(name),
            _normalize_text(identifier),
            _normalize_text(slug),
        ]
        if any(c and (norm_author == c or norm_author in c or c in norm_author) for c in candidates):
            return entry
    return None


def extract_facebook_slug(url: Optional[str]) -> Optional[str]:
    """Extract the page slug from a Facebook URL."""
    if not url:
        return None
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            return parts[0]
    except Exception:
        return None
    return None


# Storage manager with Redis fallback
class StorageManager:
    """Manages Redis or in-memory storage with automatic fallback"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_storage: Dict[str, Any] = {}
        self.use_redis = False
    
    async def connect(self):
        """Attempt Redis connection, fallback to memory"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using in-memory storage")
            return
        
        try:
            self.redis_client = redis.from_url(
                config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5
            )
            await self.redis_client.ping()
            self.use_redis = True
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage")
            self.redis_client = None
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve value from storage"""
        if self.use_redis and self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    import json
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        return self.memory_storage.get(key)
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """Store value with TTL"""
        if self.use_redis and self.redis_client:
            try:
                import json
                await self.redis_client.set(key, json.dumps(value), ex=ttl)
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        self.memory_storage[key] = value
    
    async def delete(self, key: str):
        """Delete value from storage"""
        if self.use_redis and self.redis_client:
            try:
                await self.redis_client.delete(key)
                return
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        self.memory_storage.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if self.use_redis and self.redis_client:
            try:
                return await self.redis_client.exists(key) > 0
            except Exception as e:
                logger.error(f"Redis exists error: {e}")
        
        return key in self.memory_storage


storage = StorageManager()


# Metrics tracker
class Metrics:
    """Track service metrics"""
    
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        self.active_jobs = 0
        self.start_time = time.time()
    
    def record_request(self, success: bool, processing_time: float):
        """Record request outcome"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_processing_time += processing_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics"""
        uptime = time.time() - self.start_time
        avg_time = self.total_processing_time / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": f"{(self.successful_requests / self.total_requests * 100):.1f}%" if self.total_requests > 0 else "N/A",
            "average_processing_time": f"{avg_time:.2f}s",
            "active_jobs": self.active_jobs,
            "uptime_seconds": int(uptime)
        }


metrics = Metrics()


class RateLimiter:
    """Rate limiting để tránh spam requests"""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.requests = defaultdict(list)  # IP -> list of timestamps
    
    def is_allowed(self, client_ip: str) -> tuple[bool, Optional[str]]:
        """Check if request is allowed"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.max_requests:
            wait_time = 60 - (now - self.requests[client_ip][0])
            return False, f"Rate limit exceeded. Try again in {int(wait_time)}s"
        
        # Add new request
        self.requests[client_ip].append(now)
        return True, None
    
    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for IP"""
        now = time.time()
        minute_ago = now - 60
        
        recent = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        return max(0, self.max_requests - len(recent))


rate_limiter = RateLimiter(max_requests_per_minute=60)


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("=" * 60)
    logger.info(f"Starting {config.TITLE} v{config.VERSION}")
    logger.info("=" * 60)
    
    config.validate()
    global DATASETS
    DATASETS = load_datasets(config.DATA_DIR)
    app.state.datasets = DATASETS
    if realtime_verifier:
        realtime_verifier.update_datasets(DATASETS)
    await storage.connect()
    
    # Test crawler connection
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.CRAWLER_API_URL}/health")
            if response.status_code == 200:
                logger.info(f"Crawler API connected: {config.CRAWLER_API_URL}")
            else:
                logger.warning(f"Crawler API unhealthy: {response.status_code}")
    except Exception as e:
        logger.error(f"Crawler API unreachable: {e}")
    
    logger.info("=" * 60)
    logger.info("Service ready")
    logger.info("=" * 60)
    
    yield
    
    logger.info("Shutting down...")
    await storage.close()
    logger.info("Shutdown complete")


# FastAPI application
app = FastAPI(
    title=config.TITLE,
    version=config.VERSION,
    description=config.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG") else "An error occurred"
        }
    )


# Initialize verification engines
realtime_verifier = None
USE_REALTIME = False
llm_adapter = None

# Initialize shared offline LLM (optional, fallback to heuristics)
try:
    llm_adapter = LLMAdapter()
    logger.info("LLM adapter initialized")
except Exception as exc:
    reason = str(exc)
    if isinstance(exc, LLMFallbackError):
        reason = f"LLM fallback: {exc}"
    logger.warning("LLM adapter disabled: %s", reason)
    llm_adapter = None

try:
    from trustscore.realtime_verifier import RealtimeVerifier
    
    # Get API keys from environment
    api_keys = {
        'google_factcheck': os.getenv('FACT_CHECK_API_KEY', ''),
    }
    
    realtime_verifier = RealtimeVerifier(api_keys=api_keys, llm_adapter=llm_adapter)
    logger.info("Realtime verifier initialized")
    USE_REALTIME = True
    
except ImportError as e:
    logger.error(f"Failed to load realtime verifier: {e}")

# Fallback: Original engine
engine = TrustEngine(
    aggregator=SourceAggregator(),
    reputation_client=ReputationClient(),
    semantic_analyzer=SemanticAnalyzer(llm=llm_adapter),
    language_scorer=LanguageRiskScorer(llm=llm_adapter),
    image_verifier=ImageVerifier(),
    llm=llm_adapter,
)
logger.info("Legacy trust engine initialized as fallback")


# Request/Response models
class VerifyRequest(BaseModel):
    """Verification request model"""
    
    text: str = Field(..., min_length=10, max_length=10000, description="Content to verify")
    url: Optional[str] = Field(None, description="Source URL if available")
    language: str = Field("vi", description="Language: vi or en")
    deep_analysis: bool = Field(True, description="Enable deep analysis (slower but more accurate)")
    author: Optional[str] = Field(None, description="Author or page name if available")
    found_articles: Optional[List[Dict[str, Any]]] = Field(None, description="Prefilled found articles (from extension)")
    meta: Optional[Dict[str, Any]] = Field(None, description="Additional metadata from client")
    image_urls: Optional[List[str]] = Field(None, description="Image URLs attached to the content")
    
    @validator('language')
    def validate_language(cls, v):
        if v not in ['vi', 'en']:
            raise ValueError('Language must be vi or en')
        return v
    
    @validator('text')
    def validate_text(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Text must be at least 10 characters')
        return v.strip()


class VerifyResponse(BaseModel):
    """Verification response model"""
    
    job_id: str
    status: str
    trust_score: Optional[float] = None
    verdict: Optional[str] = None
    summary: Optional[str] = None
    evidence_count: Optional[int] = None
    crawl_stats: Optional[Dict[str, Any]] = None
    components: Optional[Dict[str, float]] = None
    alternatives: Optional[list] = None
    is_donation_post: Optional[bool] = None  # NEW: Flag for donation posts
    processing_time: Optional[float] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    flags: Optional[list] = None
    evidence: Optional[list] = None
    confidence_estimate: Optional[float] = None
    override_reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# API endpoints
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": config.TITLE,
        "version": config.VERSION,
        "status": "operational",
        "endpoints": {
            "verify": "POST /verify",
            "result": "GET /result/{job_id}",
            "health": "GET /health",
            "metrics": "GET /metrics"
        },
        "external_services": {
            "crawler_url": config.CRAWLER_API_URL,
            "storage": "redis" if storage.use_redis else "memory"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    
    crawler_healthy = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.CRAWLER_API_URL}/health")
            crawler_healthy = response.status_code == 200
    except:
        pass
    
    storage_healthy = storage.use_redis
    if storage.use_redis:
        try:
            await storage.redis_client.ping()
        except:
            storage_healthy = False
    
    overall_status = "healthy" if crawler_healthy else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "healthy",
            "crawler": "healthy" if crawler_healthy else "unhealthy",
            "storage": "redis" if storage_healthy else "memory",
            "ai_engine": "loaded"
        },
        "metrics": metrics.get_stats()
    }


@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return {
        "service": config.TITLE,
        "version": config.VERSION,
        "metrics": metrics.get_stats(),
        "storage": {
            "type": "redis" if storage.use_redis else "memory",
            "memory_items": len(storage.memory_storage)
        }
    }


@app.post("/verify", response_model=VerifyResponse, status_code=status.HTTP_202_ACCEPTED)
async def verify_claim(request_body: VerifyRequest, background_tasks: BackgroundTasks, http_request: Request):
    """
    Verify a claim with AI fact-checking
    Returns job_id immediately, processing in background
    """
    
    # Rate limiting
    client_ip = http_request.client.host
    allowed, error_msg = rate_limiter.is_allowed(client_ip)
    
    if not allowed:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=error_msg
        )
    
    # Check concurrent jobs limit
    if metrics.active_jobs >= config.MAX_CONCURRENT_JOBS:
        logger.warning(f"Max concurrent jobs reached: {metrics.active_jobs}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server busy. Active jobs: {metrics.active_jobs}/{config.MAX_CONCURRENT_JOBS}. Try again later"
        )
    
    # Check cache first (same content recently verified)
    content_hash = hashlib.sha256(request_body.text.encode()).hexdigest()[:16]
    cached_result = await storage.get(f"cache:{content_hash}")
    
    if cached_result and cached_result.get("status") == "completed":
        logger.info(f"Cache hit for content: {request_body.text[:30]}...")
        cached_result["from_cache"] = True
        return VerifyResponse(**cached_result)
    
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    
    logger.info(f"[{job_id}] New verification from {client_ip}: {request_body.text[:50]}...")
    logger.info(f"[{job_id}] Remaining requests for this IP: {rate_limiter.get_remaining(client_ip)}")
    
    initial_status = VerifyResponse(
        job_id=job_id,
        status="processing",
        summary="Processing... Please wait"
    )
    
    await storage.set(
        f"job:{job_id}",
        initial_status.dict(),
        ttl=config.REDIS_TTL_JOBS
    )
    
    metrics.active_jobs += 1
    background_tasks.add_task(process_verification, job_id, request_body, content_hash)
    
    return initial_status


@app.get("/result/{job_id}", response_model=VerifyResponse)
async def get_verification_result(job_id: str):
    """Get verification result by job ID"""
    
    result = await storage.get(f"job:{job_id}")
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job ID not found or expired"
        )
    
    return VerifyResponse(**result)


@app.delete("/result/{job_id}")
async def delete_result(job_id: str):
    """Delete verification result"""
    
    if not await storage.exists(f"job:{job_id}"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job ID not found"
        )
    
    await storage.delete(f"job:{job_id}")
    logger.info(f"[{job_id}] Result deleted")
    
    return {"message": "Result deleted successfully"}


@app.post("/verify-simple", response_model=TrustScoreResult)
async def verify_simple(payload: ClaimPayload):
    """Direct verification with pre-provided documents (legacy endpoint)"""
    try:
        result = await engine.verify_claim(payload)
        return result
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


async def finalize_whitelist_result(
    job_id: str,
    request: VerifyRequest,
    domain: str,
    start_time: float,
    content_hash: Optional[str],
    reason: str = "domain_whitelist",
    trusted_page: Optional[Dict[str, Any]] = None,
    use_realtime: bool = True
):
    """
    Short-circuit pipeline for whitelisted domains: skip crawler and return authority-first result.
    """
    meta_payload = {
        "domain": domain,
        "url": request.url,
        "domain_frequency": {domain: 1},
        "first_seen": datetime.utcnow().isoformat(),
        "author": request.author,
        "trusted_page": trusted_page,
    }
    crawl_stats = {
        "skipped": True,
        "reason": reason,
        "domain": domain,
        "trusted_page": trusted_page,
    }

    trust_score = 100.0
    verdict = "verified"
    if reason == "trusted_page" and trusted_page:
        summary = f"Trusted Facebook page ({trusted_page.get('name') or request.author}) in whitelist; crawler skipped."
    else:
        summary = f"Trusted source domain ({domain}) in whitelist; crawler skipped."
    components: Dict[str, float] = {"domain_reliability": 1.0}
    flags: List[str] = ["authority-whitelist"]
    evidence_list: List[str] = [
        f"Domain in whitelist: {domain}"
        if reason != "trusted_page"
        else f"Trusted page: {trusted_page.get('identifier') or request.author}"
    ]
    confidence_estimate = None
    override_reason = "authority-whitelist"
    details: Dict[str, Any] = {"meta": meta_payload, "fast_path": reason}
    is_donation_post = False

    if USE_REALTIME and realtime_verifier and use_realtime:
        verification_result = await realtime_verifier.verify(
            original_text=request.text,
            found_articles=[],
            language=request.language,
            meta=meta_payload
        )
        trust_score = verification_result.get("trust_score", trust_score)
        verdict = verification_result.get("verdict", verdict)
        summary = verification_result.get("explanation", summary)
        components = verification_result.get("components", components) or components
        flags = verification_result.get("flags", flags) or flags
        evidence_list = verification_result.get("evidence", evidence_list) or evidence_list
        confidence_estimate = verification_result.get("confidence_estimate", confidence_estimate)
        override_reason = verification_result.get("override_reason", override_reason) or override_reason
        details = verification_result.get("details", details) or details
        is_donation_post = verification_result.get("is_donation_post", is_donation_post)

    if isinstance(details, dict):
        details.setdefault("meta", meta_payload)
        details["fast_path"] = reason
    else:
        details = {"meta": meta_payload, "fast_path": reason}

    await finalize_job(
        job_id,
        trust_score=trust_score,
        verdict=verdict,
        summary=summary,
        evidence_count=len(evidence_list) or 1,
        crawl_stats=crawl_stats,
        components=components,
        alternatives=[],
        is_donation_post=is_donation_post,
        processing_time=time.time() - start_time,
        success=True,
        content_hash=content_hash,
        flags=flags,
        evidence=evidence_list,
        confidence_estimate=confidence_estimate,
        override_reason=override_reason,
        details=details
    )


# Background processing
async def process_verification(job_id: str, request: VerifyRequest, content_hash: str = None):
    """Background task for verification processing"""
    
    start_time = time.time()
    
    try:
        logger.info(f"[{job_id}] Starting verification")

        domain_whitelist = DATASETS.get("domain_whitelist", [])
        trusted_sources = DATASETS.get("trusted_sources", [])
        logger.warning(f"[{job_id}] Debug input url={request.url}, author={request.author}")
        request_domain = extract_domain(request.url)
        if not request_domain and request.found_articles:
            request_domain = extract_domain(request.found_articles[0].get("url"))
        if not request_domain and request.meta and isinstance(request.meta, dict):
            meta_domain = request.meta.get("domain") or next(iter((request.meta.get("domain_frequency") or {}).keys()), None)
            request_domain = extract_domain(meta_domain)

        # Precompute trusted Facebook page match
        slug = extract_facebook_slug(request.url) or (
            extract_facebook_slug(request.found_articles[0].get("url")) if request.found_articles else None
        )
        trusted_pages = [s for s in trusted_sources if s.get("type") == "page"]
        page_hit = match_trusted_page(request.author, trusted_sources) or match_trusted_page(slug, trusted_sources)
        logger.warning(f"[{job_id}] Debug domain={request_domain}, slug={slug}, page_hit={bool(page_hit)}")

        # Whitelist: domain
        if request_domain and is_domain_whitelisted(request_domain, domain_whitelist):
            logger.info(f"[{job_id}] Whitelisted domain detected ({request_domain}). Skipping crawler.")
            await finalize_whitelist_result(job_id, request, request_domain, start_time, content_hash)
            return

        # Whitelist: trusted Facebook page (author or slug)
        is_facebook_source = (
            (request_domain and "facebook" in request_domain)
            or ("facebook.com" in (request.url or ""))
            or (slug is not None)
        )
        if is_facebook_source:
            if page_hit:
                logger.info(
                    f"[{job_id}] Trusted Facebook page detected ({page_hit.get('name')}). Slug={slug}, author={request.author}"
                )
                await finalize_whitelist_result(
                    job_id,
                    request,
                    request_domain or "facebook.com",
                    start_time,
                    content_hash,
                    reason="trusted_page",
                    trusted_page=page_hit,
                    use_realtime=False
                )
                return
            else:
                logger.warning(
                    f"[{job_id}] Facebook source but not trusted. author={request.author}, slug={slug}, domain={request_domain}"
                )
        
        # Call crawler API
        logger.info(f"[{job_id}] Calling crawler API")
        crawl_data = await call_crawler_api(job_id, request)
        
        if not crawl_data:
            raise Exception("Crawler API returned no data")
        
        found_articles = crawl_data.get("results", [])
        
        if not found_articles:
            await finalize_job(
                job_id,
                trust_score=20.0,
                verdict="insufficient-evidence",
                summary="Insufficient evidence found",
                evidence_count=0,
                crawl_stats=crawl_data.get("stats", {}),
                components={},
                alternatives=[],
                is_donation_post=False,
                processing_time=time.time() - start_time,
                success=False,
                content_hash=content_hash
            )
            return
        
        logger.info(f"[{job_id}] Found {len(found_articles)} articles for analysis")
        domain_counts = Counter([article.get("domain") for article in found_articles if article.get("domain")])
        crawl_data.setdefault("stats", {})["domain_frequency"] = dict(domain_counts)
        meta_payload = {
            "domain_frequency": dict(domain_counts),
            "share_count": sum(article.get("share_count", 0) or 0 for article in found_articles),
            "social_signals": crawl_data.get("stats", {}).get("social_signals"),
            "user_reports": crawl_data.get("stats", {}).get("user_reports"),
            "image_urls": [
                url
                for article in found_articles
                for url in (article.get("image_urls") or [])
            ],
        }
        
        # Use RealtimeVerifier if available
        if USE_REALTIME:
            logger.info(f"[{job_id}] Running realtime verification")
            
            verification_result = await realtime_verifier.verify(
                original_text=request.text,
                found_articles=found_articles,
                language=request.language,
                meta=meta_payload
            )
            
            trust_score = verification_result.get("trust_score", 0)
            verdict = verification_result.get("verdict", "needs-review")
            explanation = verification_result.get("explanation", "")
            components = verification_result.get("components", {})
            flags = verification_result.get("flags", [])
            evidence_list = verification_result.get("evidence", [])
            details = verification_result.get("details", {})
            confidence_estimate = verification_result.get("confidence_estimate")
            override_reason = verification_result.get("override_reason")
            
            logger.info(f"[{job_id}] Verification completed. Score: {trust_score}/100")
            
            # Extract alternatives from authority check
            alternatives = []
            domain_whitelist = DATASETS.get("domain_whitelist", [])
            for article in found_articles[:5]:
                domain = article.get("domain", "")
                if article.get("url_trust") or any(d.lower() in domain.lower() for d in domain_whitelist):
                    alternatives.append({
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "source": domain
                    })
            
            await finalize_job(
                job_id,
                trust_score=trust_score,
                verdict=verdict,
                summary=explanation,
                evidence_count=len(evidence_list) or len(found_articles),
                crawl_stats=crawl_data.get("stats", {}),
                components=components,
                alternatives=alternatives,
                is_donation_post=verification_result.get("is_donation_post", False),
                processing_time=time.time() - start_time,
                success=True,
                content_hash=content_hash,
                flags=flags,
                evidence=evidence_list,
                confidence_estimate=confidence_estimate,
                override_reason=override_reason,
                details=details
            )
            
        else:
            # Fallback to legacy engine
            logger.info(f"[{job_id}] Using legacy verification engine")
            
            documents = prepare_documents(crawl_data, request.language)
            
            claim_payload = ClaimPayload(
                claim_text=request.text,
                url=request.url,
                documents=documents,
                language=request.language,
                context=request.text
            )
            
            trust_result = await engine.verify_claim(claim_payload)
            
            logger.info(f"[{job_id}] Verification completed. Score: {trust_result.trust_score:.2f}")
            
            verdict_summaries = {
                "verified": "Information appears to be accurate based on multiple trusted sources",
                "needs-review": "Information requires further review",
                "likely-false": "Information is likely false or unverified",
                "insufficient-evidence": "Insufficient evidence to assess"
            }
            
            await finalize_job(
                job_id,
                trust_score=round(trust_result.trust_score * 100, 1),
                verdict=trust_result.verdict,
                summary=verdict_summaries.get(trust_result.verdict, ""),
                evidence_count=len(trust_result.evidence),
                crawl_stats=crawl_data.get("stats", {}),
                components={k: round(v * 100, 1) for k, v in trust_result.components.items()},
                alternatives=[
                    {"title": alt.title, "url": str(alt.url), "source": alt.source}
                    for alt in trust_result.alternatives[:5]
                ],
                is_donation_post=False,
                processing_time=time.time() - start_time,
                success=True,
                content_hash=content_hash
            )
        
    except Exception as e:
        logger.error(f"[{job_id}] Verification failed: {e}", exc_info=True)
        
        processing_time = time.time() - start_time
        
        await storage.set(
            f"job:{job_id}",
            {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "processing_time": round(processing_time, 2),
                "completed_at": datetime.utcnow().isoformat()
            },
            ttl=config.REDIS_TTL_JOBS
        )
        
        metrics.record_request(success=False, processing_time=processing_time)
        metrics.active_jobs -= 1


async def call_crawler_api(job_id: str, request: VerifyRequest) -> Dict[str, Any]:
    """Call crawler API to retrieve data"""
    
    base_url = config.CRAWLER_API_URL.rstrip("/")
    analyze_url = f"{base_url}/api/article/analyze"
    search_url = f"{base_url}/search"

    crawler_request = {
        "query": request.text[:200],
        "max_results": 30 if request.deep_analysis else 15,
        "include_sources": ["google", "news", "government"],
        "languages": [request.language],
        "deep_crawl": request.deep_analysis,
        "timeout": 120
    }
    
    async with httpx.AsyncClient(timeout=config.CRAWLER_TIMEOUT) as client:
        try:
            analyze_payload = build_analyze_payload(request)
            logger.info(f"[{job_id}] Posting to crawler analyze endpoint: {analyze_url}")
            response = await client.post(
                analyze_url,
                json=analyze_payload
            )
            response.raise_for_status()

            raw = response.json()
            adapted = adapt_analyze_response(raw, request.language)
            logger.info(
                f"[{job_id}] Crawler returned {len(adapted.get('results', []))} normalized results "
                f"(source total: {len(raw.get('data', []))})"
            )
            return adapted

        except httpx.TimeoutException:
            logger.error(f"[{job_id}] Crawler API timeout")
            raise Exception("Crawler API timeout")
        except httpx.HTTPError as e:
            logger.error(f"[{job_id}] Crawler analyze error: {e}")
        except Exception as e:
            logger.error(f"[{job_id}] Unexpected crawler analyze error: {e}")

        # Nếu analyze fail, trả về dữ liệu trống để hệ thống kết luận "insufficient-evidence" thay vì nổ job
        logger.warning(f"[{job_id}] Returning empty crawler data due to analyze failure")
        return {"results": [], "stats": {}, "main_search": None}


def build_analyze_payload(request: VerifyRequest) -> Dict[str, Any]:
    """Build payload for crawler /api/article/analyze endpoint"""

    title_hint = request.text.strip().split("\n")[0][:140] if request.text else ""

    return {
        "url": request.url or "about:blank",
        "title": title_hint or request.url or "",
        "article": request.text,
        "created_at": None,
        "author": None,
        "platform": "web",
        "image_urls": [],
    }


def adapt_analyze_response(raw: Dict[str, Any], language: str) -> Dict[str, Any]:
    """
    Adapt crawler /api/article/analyze response to the model's expected schema.
    Ensures keys: results (list[dict]), stats (dict)
    """
    if not raw or "data" not in raw:
        return raw or {}

    meta = raw.get("meta") or {}
    related_articles = raw.get("data") or []

    results = []
    for article in related_articles:
        body = article.get("article") or article.get("content") or ""
        snippet = body[:280] + ("..." if len(body) > 280 else "")
        results.append({
            "url": article.get("url"),
            "title": article.get("title"),
            "content": body or None,
            "snippet": snippet or article.get("title"),
            "domain": article.get("domain"),
            "published_time": article.get("created_at"),
            "author": article.get("author"),
            "platform": article.get("platform") or "web",
            "image_urls": article.get("image_urls"),
            "share_count": article.get("share_count", 0),
            "url_trust": article.get("is_verified_account", False),
            "language": language,
            "trust_score": article.get("trust_score", 50),
        })

    stats = {
        "total_found": meta.get("total", len(related_articles)),
        "page": meta.get("page"),
        "has_next": meta.get("has_next"),
        "domain_frequency": meta.get("domain_frequency") or {},
        "source_platforms": dict(Counter([r.get("platform") for r in results if r.get("platform")])),
    }

    return {
        "results": results,
        "stats": stats,
        "main_search": raw.get("main_search"),
    }


def prepare_documents(crawl_data: Dict[str, Any], language: str) -> list:
    """Prepare documents from crawler data for AI engine"""
    
    documents = []
    
    for result in crawl_data.get("results", []):
        if not (result.get("content") or result.get("snippet")):
            continue
        
        doc = {
            "id": result.get("url", f"doc_{len(documents)}"),
            "role": "support",
            "title": result.get("title"),
            "url": result.get("url"),
            "source": result.get("domain"),
            "content": result.get("content"),
            "excerpt": result.get("snippet"),
            "published_at": result.get("published_time"),
            "verified_origin": result.get("url_trust", False),
            "confidence_hint": result.get("trust_score", 50) / 100.0,
            "language": result.get("language", language)
        }
        
        documents.append(doc)
    
    return documents


async def finalize_job(
    job_id: str,
    trust_score: float,
    verdict: str,
    summary: str,
    evidence_count: int,
    crawl_stats: Dict[str, Any],
    components: Dict[str, float],
    alternatives: list,
    is_donation_post: bool,
    processing_time: float,
    success: bool,
    content_hash: str = None,
    flags: Optional[list] = None,
    evidence: Optional[list] = None,
    confidence_estimate: Optional[float] = None,
    override_reason: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
):
    """Finalize and store job result"""
    
    result = {
        "job_id": job_id,
        "status": "completed",
        "trust_score": trust_score,
        "verdict": verdict,
        "summary": summary,
        "evidence_count": evidence_count,
        "crawl_stats": crawl_stats,
        "components": components,
        "alternatives": alternatives,
        "is_donation_post": is_donation_post,
        "processing_time": round(processing_time, 2),
        "completed_at": datetime.utcnow().isoformat()
    }
    
    if flags is not None:
        result["flags"] = flags
    if evidence is not None:
        result["evidence"] = evidence
    if confidence_estimate is not None:
        result["confidence_estimate"] = confidence_estimate
    if override_reason:
        result["override_reason"] = override_reason
    if details is not None:
        result["details"] = details
    
    # Store by job_id
    await storage.set(
        f"job:{job_id}",
        result,
        ttl=config.REDIS_TTL_JOBS
    )
    
    # Also cache by content hash (for duplicate requests)
    if content_hash:
        await storage.set(
            f"cache:{content_hash}",
            result,
            ttl=config.REDIS_TTL_CACHE
        )
    
    metrics.record_request(success=success, processing_time=processing_time)
    metrics.active_jobs -= 1
    
    logger.info(f"[{job_id}] Job finalized. Score: {trust_score}/100, Time: {processing_time:.2f}s")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
