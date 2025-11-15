"""
Improved Analyzers với NLI Model và Vietnamese Support
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TYPE_CHECKING

import httpx

from .config import get_settings

# Try import advanced models
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ transformers not available. Using fallback semantic analyzer")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class AdvancedSemanticAnalyzer:
    """
    Advanced semantic analyzer với NLI (Natural Language Inference)
    Hỗ trợ tiếng Việt và tiếng Anh
    """
    
    def __init__(self, *, model_name: str | None = None, max_workers: int = 2) -> None:
        settings = get_settings()
        
        # NLI model for fact-checking (multilingual)
        self._nli_model_name = model_name or "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        
        # Fallback: sentence transformer
        self._st_model_name = settings.semantic_model_name
        
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._nli_model = None
        self._nli_tokenizer = None
        self._st_model = None
        
        # Load models lazily
        self._models_loaded = False
    
    def _ensure_models(self):
        """Load models if not loaded yet"""
        if self._models_loaded:
            return
        
        try:
            if TRANSFORMERS_AVAILABLE:
                print(f"Loading NLI model: {self._nli_model_name}")
                self._nli_tokenizer = AutoTokenizer.from_pretrained(self._nli_model_name)
                self._nli_model = AutoModelForSequenceClassification.from_pretrained(self._nli_model_name)
                self._nli_model.eval()  # Set to eval mode
                print("✅ NLI model loaded")
            else:
                print("⚠️ Transformers not available, using fallback")
        except Exception as e:
            print(f"⚠️ Failed to load NLI model: {e}")
            self._nli_model = None
        
        # Fallback to sentence transformer
        if not self._nli_model and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading Sentence Transformer: {self._st_model_name}")
                self._st_model = SentenceTransformer(self._st_model_name)
                print("✅ Sentence Transformer loaded")
            except Exception as e:
                print(f"⚠️ Failed to load Sentence Transformer: {e}")
        
        self._models_loaded = True
    
    async def score(self, claim_text: str, references: Sequence[str]) -> float:
        """
        Score claim against references using NLI or semantic similarity
        
        Returns: float between -1 and 1
            1 = strong support (entailment)
            0 = neutral
           -1 = contradiction
        """
        texts = [text for text in references if text]
        if not claim_text or not texts:
            return 0.0
        
        self._ensure_models()
        
        # Use NLI if available
        if self._nli_model and self._nli_tokenizer:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                self._compute_nli_score,
                claim_text,
                texts
            )
        
        # Fallback to sentence transformer
        elif self._st_model:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                self._compute_similarity,
                self._st_model,
                claim_text,
                texts
            )
        
        # No models available
        else:
            return 0.5  # Neutral
    
    def _compute_nli_score(self, claim_text: str, references: Sequence[str]) -> float:
        """
        Compute NLI score: entailment vs contradiction
        
        NLI Labels:
            0: contradiction
            1: neutral
            2: entailment
        """
        if not self._nli_model or not self._nli_tokenizer:
            return 0.5
        
        scores = []
        
        for reference in references[:10]:  # Limit to 10 refs for performance
            try:
                # Tokenize
                inputs = self._nli_tokenizer(
                    claim_text,
                    reference,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # Forward pass
                with torch.no_grad():
                    outputs = self._nli_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)[0]
                
                # Extract probabilities
                contradiction_prob = probs[0].item()
                neutral_prob = probs[1].item()
                entailment_prob = probs[2].item()
                
                # Compute score: entailment is good, contradiction is bad
                # Score = entailment - contradiction
                score = entailment_prob - contradiction_prob
                scores.append(score)
                
            except Exception as e:
                print(f"NLI error: {e}")
                continue
        
        if not scores:
            return 0.0
        
        # Average score
        avg_score = sum(scores) / len(scores)
        return round(float(avg_score), 3)
    
    @staticmethod
    def _compute_similarity(model, claim_text: str, texts: Sequence[str]) -> float:
        """Fallback: compute cosine similarity using sentence transformers"""
        try:
            embeddings = model.encode([claim_text, *texts], convert_to_tensor=True, normalize_embeddings=True)
            claim_embedding = embeddings[0]
            reference_embeddings = embeddings[1:]
            similarities = util.dot_score(claim_embedding, reference_embeddings).tolist()[0]
            best_match = max(similarities, default=0.0)
            return round(float(best_match), 3)
        except Exception as e:
            print(f"Similarity error: {e}")
            return 0.5


class ImprovedLanguageRiskScorer:
    """
    Improved language risk scorer với Vietnamese clickbait patterns
    """
    
    VIETNAMESE_CLICKBAIT_PATTERNS = (
        r"!{2,}",
        r"\b(?:giật gân|sốc|chấn động|bạn sẽ không tin|không ai ngờ|gây sốt|hot)\b",
        r"\b(?:100%|cam kết|đảm bảo|chưa từng có|độc nhất|duy nhất)\b",
        r"\b(?:khó tin|không thể tin|choáng váng|bất ngờ|kinh hoàng)\b",
        r"\b(?:tiết lộ|phanh phui|bí mật|vạch trần)\b",
    )
    
    ENGLISH_CLICKBAIT_PATTERNS = (
        r"\b(?:shocking|unbelievable|you won't believe|mind-blowing|incredible)\b",
        r"\b(?:secret|revealed|exposed|truth about)\b",
        r"\b(?:100%|guaranteed|proven|never before|exclusive)\b",
        r"\b(?:click here|find out|discover|learn more)\b",
    )
    
    ALL_CAPS_PATTERN = re.compile(r"[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]{5,}")
    
    def score(self, text: str, *, language: str | None = None) -> float:
        """
        Score language risk (0 = high risk, 1 = credible)
        """
        if not text:
            return 0.5
        
        penalties = 0.0
        word_count = max(len(text.split()), 1)
        
        # Check excessive punctuation
        exclamation_ratio = text.count("!") / word_count
        if exclamation_ratio > 0.05:
            penalties += 0.15
        
        question_ratio = text.count("?") / word_count
        if question_ratio > 0.1:
            penalties += 0.1
        
        # Check ALL CAPS
        if self.ALL_CAPS_PATTERN.search(text):
            penalties += 0.2
        
        # Check clickbait patterns
        patterns = self.VIETNAMESE_CLICKBAIT_PATTERNS
        if language and language.lower().startswith("en"):
            patterns = self.ENGLISH_CLICKBAIT_PATTERNS
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                penalties += 0.15 * min(len(matches), 3)  # Max 3 matches
        
        # Check emotional words density
        emotional_words_vi = ["ghê", "khủng", "kinh", "sợ", "phẫn nộ", "tức giận"]
        emotional_words_en = ["terrible", "horrible", "amazing", "awesome", "hate", "love"]
        
        emotional_words = emotional_words_vi if language == "vi" else emotional_words_en
        emotional_count = sum(text.lower().count(word) for word in emotional_words)
        
        if emotional_count / word_count > 0.03:  # > 3% emotional words
            penalties += 0.1
        
        # Vietnamese gets slightly higher penalty for clickbait
        if language and language.lower().startswith("vi"):
            penalties *= 1.1
        
        # Compute credibility score
        credible_score = max(0.0, 1.0 - min(0.8, penalties))
        return round(credible_score, 3)


class ImprovedImageVerifier:
    """
    Improved image verifier với better fallback
    """
    
    def __init__(self, *, timeout: float = 12.0) -> None:
        settings = get_settings()
        self._timeout = timeout
        self._endpoint = settings.image_search_api_url
        self._api_key = settings.image_search_api_key
    
    async def score(self, media_hashes: Sequence[str]) -> float:
        """
        Verify images using reverse image search
        Returns confidence 0-1
        """
        if not media_hashes:
            return 0.6  # Neutral if no images
        
        if not self._endpoint or not self._api_key:
            # No API configured - can't verify, return neutral
            return 0.55
        
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [self._query_hash(client, media_hash) for media_hash in media_hashes[:5]]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        scores = [response for response in responses if isinstance(response, float)]
        if not scores:
            return 0.55  # No results, neutral
        
        return round(sum(scores) / len(scores), 3)
    
    async def _query_hash(self, client: httpx.AsyncClient, media_hash: str) -> float:
        """Query reverse image search API"""
        payload = {"hash": media_hash}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            response = await client.post(self._endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return float(data.get("confidence", 0.5))
        except httpx.HTTPError:
            return 0.5

