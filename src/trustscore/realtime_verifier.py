"""
Realtime Verification Engine
Rule-based + heuristics approach for fake news detection
"""

import re
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import hashlib


def load_spam_patterns() -> Dict[str, Any]:
    """Load spam patterns from JSON file"""
    config_path = Path(__file__).parent.parent.parent / "config" / "spam_patterns.json"
    
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Fallback to defaults
    return {
        "vietnamese": {"clickbait_phrases": [], "fake_indicators": []},
        "english": {"clickbait_phrases": [], "fake_indicators": []}
    }


class SpamPatterns:
    """Spam and clickbait patterns for Vietnamese and English"""
    
    def __init__(self):
        patterns_data = load_spam_patterns()
        
        self.VIETNAMESE_SPAM = patterns_data["vietnamese"]["clickbait_phrases"]
        self.VIETNAMESE_FAKE = patterns_data["vietnamese"]["fake_indicators"]
        self.VIETNAMESE_EMOTIONAL = patterns_data["vietnamese"].get("emotional_words", [])
        
        self.ENGLISH_SPAM = patterns_data["english"]["clickbait_phrases"]
        self.ENGLISH_FAKE = patterns_data["english"]["fake_indicators"]
        self.ENGLISH_EMOTIONAL = patterns_data["english"].get("emotional_words", [])
        
        self.TRUSTED_DOMAINS_VN = set(patterns_data["trusted_domains"]["vietnamese"])
        self.TRUSTED_DOMAINS_EN = set(patterns_data["trusted_domains"]["english"])
        self.SOCIAL_MEDIA = set(patterns_data["social_media"])
    
    def detect_spam_language(self, text: str, language: str = "vi") -> Dict[str, Any]:
        """Detect spam patterns in text"""
        if language == "vi":
            patterns = self.VIETNAMESE_SPAM + self.VIETNAMESE_FAKE + self.VIETNAMESE_EMOTIONAL
        else:
            patterns = self.ENGLISH_SPAM + self.ENGLISH_FAKE + self.ENGLISH_EMOTIONAL
        
        found_patterns = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_patterns.append(pattern)
        
        spam_score = min(len(found_patterns) * 0.15, 1.0)
        
        return {
            "spam_score": spam_score,
            "found_patterns": found_patterns,
            "pattern_count": len(found_patterns)
        }


class ContentDuplicationChecker:
    """Check content duplication and uniqueness"""
    
    @staticmethod
    def calculate_fingerprint(text: str) -> str:
        """Calculate content fingerprint using hash"""
        # Normalize text
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Create hash
        return hashlib.md5(normalized.encode()).hexdigest()
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate simple similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @classmethod
    def check_duplication(cls, original: str, found_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if content is duplicated in found articles"""
        
        original_fingerprint = cls.calculate_fingerprint(original)
        
        exact_matches = 0
        high_similarity = 0
        similarities = []
        
        for article in found_articles:
            content = article.get("content", "") or article.get("snippet", "")
            if not content:
                continue
            
            # Check exact match
            article_fingerprint = cls.calculate_fingerprint(content)
            if article_fingerprint == original_fingerprint:
                exact_matches += 1
                similarities.append(1.0)
                continue
            
            # Check similarity
            similarity = cls.calculate_similarity(original, content)
            similarities.append(similarity)
            
            if similarity > 0.8:
                high_similarity += 1
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        return {
            "exact_matches": exact_matches,
            "high_similarity_count": high_similarity,
            "total_compared": len(found_articles),
            "average_similarity": round(avg_similarity, 3),
            "is_unique": exact_matches == 0 and high_similarity == 0
        }


class AuthorityChecker:
    """Check if content appears on trusted sources"""
    
    def __init__(self):
        patterns_data = load_spam_patterns()
        self.TRUSTED_DOMAINS_VN = set(patterns_data["trusted_domains"]["vietnamese"])
        self.TRUSTED_DOMAINS_EN = set(patterns_data["trusted_domains"]["english"])
        self.SOCIAL_MEDIA = set(patterns_data["social_media"])
    
    def check_authority(self, found_articles: List[Dict[str, Any]], language: str = "vi") -> Dict[str, Any]:
        """Check if content appears on trusted sources"""
        
        # Combine both VN and EN trusted domains
        trusted_domains = self.TRUSTED_DOMAINS_VN.union(self.TRUSTED_DOMAINS_EN)
        
        trusted_sources = []
        social_sources = []
        unknown_sources = []
        
        for article in found_articles:
            domain = article.get("domain", "").lower()
            
            # Check if domain is in trusted list
            is_trusted = False
            for trusted in trusted_domains:
                if trusted.lower() in domain:
                    is_trusted = True
                    break
            
            # Also check url_trust flag from crawler
            if article.get("url_trust"):
                is_trusted = True
            
            if is_trusted:
                trusted_sources.append(article)
            elif any(social in domain for social in self.SOCIAL_MEDIA):
                social_sources.append(article)
            else:
                unknown_sources.append(article)
        
        has_authority = len(trusted_sources) > 0
        authority_score = len(trusted_sources) / len(found_articles) if found_articles else 0.0
        
        return {
            "has_authority": has_authority,
            "authority_score": round(authority_score, 3),
            "trusted_count": len(trusted_sources),
            "social_count": len(social_sources),
            "unknown_count": len(unknown_sources),
            "trusted_domains": [a.get("domain") for a in trusted_sources]
        }


class SpamBehaviorDetector:
    """Detect spam behavior (same content posted multiple times)"""
    
    @staticmethod
    def detect_spam_behavior(found_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect if same content is spammed across sources"""
        
        # Count by domain
        domain_counts = Counter()
        author_counts = Counter()
        
        for article in found_articles:
            domain = article.get("domain", "unknown")
            author = article.get("author", "unknown")
            
            domain_counts[domain] += 1
            if author != "unknown":
                author_counts[author] += 1
        
        # Check for spam patterns
        max_domain_count = max(domain_counts.values()) if domain_counts else 0
        max_author_count = max(author_counts.values()) if author_counts else 0
        
        is_spam = max_domain_count > 5 or max_author_count > 3
        
        spam_score = 0.0
        if max_domain_count > 10:
            spam_score = 0.9
        elif max_domain_count > 5:
            spam_score = 0.6
        elif max_author_count > 5:
            spam_score = 0.7
        elif max_author_count > 3:
            spam_score = 0.4
        
        return {
            "is_spam": is_spam,
            "spam_score": spam_score,
            "max_domain_count": max_domain_count,
            "max_author_count": max_author_count,
            "unique_domains": len(domain_counts),
            "unique_authors": len(author_counts)
        }


class RealtimeVerifier:
    """Main realtime verification engine"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.spam_patterns = SpamPatterns()
        self.duplication_checker = ContentDuplicationChecker()
        self.authority_checker = AuthorityChecker()
        self.spam_detector = SpamBehaviorDetector()
        
        # Fact extraction and external verification
        try:
            from .fact_extractor import FactVerifier
            from .external_verifier import CombinedExternalVerifier
            from .donation_detector import DonationDetector
            
            self.fact_verifier = FactVerifier()
            self.external_verifier = CombinedExternalVerifier(api_keys)
            self.donation_detector = DonationDetector()
            self.use_advanced = True
        except ImportError:
            self.use_advanced = False
    
    async def verify(
        self, 
        original_text: str, 
        found_articles: List[Dict[str, Any]],
        language: str = "vi"
    ) -> Dict[str, Any]:
        """
        Main verification logic
        
        Returns verdict based on:
        1. Spam language detection
        2. Content duplication
        3. Authority verification
        4. Spam behavior
        5. Fact extraction & verification
        6. External fact-checking APIs
        7. Donation scam detection (NEW)
        """
        
        # Check if this is a donation post first
        is_donation = False
        donation_check = None
        
        if self.use_advanced:
            is_donation = self.donation_detector.is_donation_post(original_text, language)
            
            if is_donation:
                donation_check = self.donation_detector.check_legitimacy(
                    original_text, found_articles, language
                )
        
        # 1. Check spam language
        spam_lang = self.spam_patterns.detect_spam_language(original_text, language)
        
        # 2. Check duplication
        duplication = self.duplication_checker.check_duplication(original_text, found_articles)
        
        # 3. Check authority
        authority = self.authority_checker.check_authority(found_articles, language)
        
        # 4. Check spam behavior
        spam_behavior = self.spam_detector.detect_spam_behavior(found_articles)
        
        # 5. Fact verification
        fact_check = None
        if self.use_advanced:
            fact_check = self.fact_verifier.verify_all(original_text, language)
        
        # 6. External verification
        external_check = None
        if self.use_advanced:
            urls = [a.get('url') for a in found_articles if a.get('url')]
            external_check = await self.external_verifier.verify(
                claim=original_text[:500],
                urls=urls[:10]
            )
        
        # Calculate trust score (with donation check)
        trust_score = self._calculate_trust_score(
            spam_lang, duplication, authority, spam_behavior, found_articles,
            fact_check, external_check, donation_check
        )
        
        # Determine verdict (donation scams have priority)
        verdict = self._determine_verdict(
            trust_score, spam_lang, duplication, authority, spam_behavior,
            fact_check, external_check, donation_check
        )
        
        # Generate explanation (with donation warning)
        explanation = self._generate_explanation(
            verdict, spam_lang, duplication, authority, spam_behavior,
            fact_check, external_check, donation_check
        )
        
        return {
            "trust_score": round(trust_score, 1),
            "verdict": verdict,
            "explanation": explanation,
            "is_donation_post": is_donation,
            "details": {
                "spam_language": spam_lang,
                "duplication": duplication,
                "authority": authority,
                "spam_behavior": spam_behavior,
                "fact_check": fact_check,
                "external_verification": external_check,
                "donation_verification": donation_check
            }
        }
    
    def _calculate_trust_score(
        self,
        spam_lang: Dict,
        duplication: Dict,
        authority: Dict,
        spam_behavior: Dict,
        found_articles: List,
        fact_check: Optional[Dict] = None,
        external_check: Optional[Dict] = None,
        donation_check: Optional[Dict] = None
    ) -> float:
        """Calculate trust score 0-100"""
        
        score = 50.0  # Start neutral
        
        # DONATION POSTS: Special handling
        if donation_check:
            risk_score = donation_check['risk_score']
            
            # High risk donation = very low trust
            if donation_check['is_likely_scam']:
                score = max(10, 100 - risk_score)
            else:
                # Legitimate donation
                if donation_check['legitimacy']['verified_by_gov']:
                    score = 85
                elif donation_check['legitimacy']['verified_by_news']:
                    score = 75
                else:
                    score = max(30, 100 - risk_score)
            
            # Skip other calculations for donation posts
            return max(0, min(100, score))
        
        # REGULAR POSTS: Normal calculation
        
        # Authority boost (most important)
        if authority["has_authority"]:
            score += 30 * authority["authority_score"]
        
        # Multiple sources boost
        if len(found_articles) > 5:
            score += 10
        elif len(found_articles) > 10:
            score += 15
        
        # Penalties
        
        # Spam language penalty
        score -= 20 * spam_lang["spam_score"]
        
        # Unique content penalty (only 1 source = suspicious)
        if duplication["is_unique"] and len(found_articles) < 2:
            score -= 30
        
        # Exact match penalty (100% copy)
        if duplication["exact_matches"] > 0 and not authority["has_authority"]:
            score -= 15
        
        # Spam behavior penalty
        score -= 25 * spam_behavior["spam_score"]
        
        # Fact check penalties
        if fact_check:
            if fact_check.get('total_suspicious', 0) > 0:
                penalty = min(20, fact_check['total_suspicious'] * 10)
                score -= penalty
            
            # Bonus for no suspicious facts
            if fact_check.get('total_facts', 0) > 0 and fact_check.get('total_suspicious', 0) == 0:
                score += 5
        
        # External verification boost/penalty
        if external_check:
            fact_checking = external_check.get('fact_checking', {})
            if fact_checking.get('has_external_verification'):
                external_score = fact_checking.get('external_score', 0.5)
                # Strong signal from fact-checkers
                if external_score > 0.8:
                    score += 20  # Verified by fact-checkers
                elif external_score < 0.3:
                    score -= 25  # Debunked by fact-checkers
        
        return max(0, min(100, score))
    
    def _determine_verdict(
        self,
        trust_score: float,
        spam_lang: Dict,
        duplication: Dict,
        authority: Dict,
        spam_behavior: Dict,
        fact_check: Optional[Dict] = None,
        external_check: Optional[Dict] = None,
        donation_check: Optional[Dict] = None
    ) -> str:
        """Determine verdict based on analysis"""
        
        # DONATION POSTS: Priority handling
        if donation_check:
            if donation_check['is_likely_scam']:
                return "donation-scam"  # Special verdict for scams
            elif donation_check['legitimacy']['verified_by_gov']:
                return "verified"
            elif donation_check['legitimacy']['verified_by_news']:
                return "needs-review"  # Safer but still review
            else:
                return "needs-review"  # Always review donations
        
        # External fact-checkers have highest priority
        if external_check:
            fact_checking = external_check.get('fact_checking', {})
            if fact_checking.get('has_external_verification'):
                google = fact_checking.get('google_factcheck', {})
                
                if google.get('found') and google.get('verdict') == 'false':
                    return "likely-false"
                elif google.get('found') and google.get('verdict') == 'true':
                    return "verified"
        
        # Red flags for fake
        if spam_behavior["is_spam"]:
            return "likely-false"
        
        if duplication["is_unique"] and not authority["has_authority"]:
            return "likely-false"
        
        if spam_lang["spam_score"] > 0.5:
            return "likely-false"
        
        # Fact check red flags
        if fact_check and fact_check.get('total_suspicious', 0) >= 2:
            return "likely-false"
        
        # Green flags for real
        if authority["has_authority"] and authority["trusted_count"] > 2:
            return "verified"
        
        # Based on trust score
        if trust_score >= 70:
            return "verified"
        elif trust_score >= 45:
            return "needs-review"
        else:
            return "likely-false"
    
    def _generate_explanation(
        self,
        verdict: str,
        spam_lang: Dict,
        duplication: Dict,
        authority: Dict,
        spam_behavior: Dict,
        fact_check: Optional[Dict] = None,
        external_check: Optional[Dict] = None,
        donation_check: Optional[Dict] = None
    ) -> str:
        """Generate human-readable explanation"""
        
        reasons = []
        
        # DONATION WARNING (highest priority)
        if donation_check:
            warning = self.donation_detector.generate_donation_warning(donation_check)
            if warning:
                reasons.append(f"DONATION POST: {warning}")
        
        # External verification
        if external_check:
            google = external_check.get('fact_checking', {}).get('google_factcheck', {})
            if google.get('found'):
                verdict_text = google.get('verdict', 'unknown')
                claim_count = google.get('claim_count', 0)
                reasons.append(f"External fact-checkers: {verdict_text} ({claim_count} sources)")
        
        if authority["has_authority"]:
            reasons.append(f"Found on {authority['trusted_count']} trusted sources")
        
        if spam_behavior["is_spam"]:
            reasons.append("Content is being spammed across multiple sources")
        
        if duplication["is_unique"]:
            reasons.append("Content only appears on single source (suspicious)")
        
        if spam_lang["pattern_count"] > 0:
            reasons.append(f"Contains {spam_lang['pattern_count']} clickbait patterns")
        
        if duplication["exact_matches"] > 0:
            reasons.append(f"Found {duplication['exact_matches']} exact duplicates")
        
        # Fact check results
        if fact_check and fact_check.get('total_suspicious', 0) > 0:
            reasons.append(f"{fact_check['total_suspicious']} suspicious facts detected")
        
        if not reasons:
            reasons.append("Mixed signals, requires manual review")
        
        return " | ".join(reasons)

