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
            from .temporal_verifier import TemporalVerifier
            
            self.fact_verifier = FactVerifier()
            self.external_verifier = CombinedExternalVerifier(api_keys)
            self.donation_detector = DonationDetector()
            self.temporal_verifier = TemporalVerifier()
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
        
        # 7. NEW: Temporal verification (old news manipulation)
        temporal_check = None
        if self.use_advanced:
            temporal_check = self.temporal_verifier.detect_old_news_manipulation(
                original_text,
                found_articles
            )
        
        # Calculate trust score (with all checks)
        trust_score = self._calculate_trust_score(
            spam_lang, duplication, authority, spam_behavior, found_articles,
            fact_check, external_check, donation_check, temporal_check
        )
        
        # Determine verdict (new thresholds)
        verdict = self._determine_verdict(
            trust_score, spam_lang, duplication, authority, spam_behavior,
            fact_check, external_check, donation_check, temporal_check
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            verdict, spam_lang, duplication, authority, spam_behavior,
            fact_check, external_check, donation_check, temporal_check
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
                "donation_verification": donation_check,
                "temporal_verification": temporal_check
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
        donation_check: Optional[Dict] = None,
        temporal_check: Optional[Dict] = None
    ) -> float:
        """
        Calculate trust score 0-100
        ALL posts go through ALL checks - no shortcuts!
        """
        
        score = 50.0  # Start neutral
        
        # === POSITIVE FACTORS (BOOST) ===
        
        # Authority boost
        if authority["has_authority"]:
            score += 40 * authority["authority_score"]
            
            if authority["trusted_count"] >= 3:
                score += 10
        
        # NEW: Individual donation with verifiable info boost
        if donation_check:
            individual = donation_check.get('individual_legitimacy')
            if individual:
                verif_score = individual.get('verifiability_score', 0)
                if verif_score >= 60:
                    # Has hospital, address, phone → boost
                    score += 25
                elif verif_score >= 30:
                    score += 15
        
        # Multiple sources boost
        if len(found_articles) > 5:
            score += 5
        elif len(found_articles) > 10:
            score += 10
        
        # External fact-checkers boost
        if external_check:
            fact_checking = external_check.get('fact_checking', {})
            if fact_checking.get('has_external_verification'):
                external_score = fact_checking.get('external_score', 0.5)
                if external_score > 0.8:
                    score += 20
                elif external_score < 0.3:
                    score -= 25
        
        # === NEGATIVE FACTORS (PENALTIES) ===
        
        # Spam language penalty
        score -= 20 * spam_lang["spam_score"]
        
        # Unique content penalty (skip for individual legitimate donations)
        if duplication["is_unique"] and len(found_articles) < 2:
            # Exception: individual donation with verifiable info
            if donation_check:
                individual = donation_check.get('individual_legitimacy')
                if individual and individual.get('legitimacy') == 'likely-legitimate':
                    # Has verifiable info → don't penalize for being unique
                    pass
                else:
                    score -= 30
            else:
                score -= 30
        
        # Exact match penalty
        if duplication["exact_matches"] > 0 and not authority["has_authority"]:
            score -= 15
        
        # Spam behavior penalty
        score -= 25 * spam_behavior["spam_score"]
        
        # Fact check penalties
        if fact_check:
            if fact_check.get('total_suspicious', 0) > 0:
                penalty = min(20, fact_check['total_suspicious'] * 10)
                score -= penalty
            
            if fact_check.get('total_facts', 0) > 0 and fact_check.get('total_suspicious', 0) == 0:
                score += 5
        
        # Temporal verification penalty
        if temporal_check:
            if temporal_check.get('is_old_news_manipulation'):
                penalty = min(30, temporal_check.get('manipulation_score', 0))
                score -= penalty
        
        # === CRITICAL PENALTIES (OVERRIDE EVERYTHING) ===
        
        # DONATION: Fake bank account detected
        if donation_check:
            account_ver = donation_check.get('account_verification', {})
            
            if account_ver.get('has_fake_accounts'):
                # Fake account = auto fail, regardless of other factors
                score = 10
            elif (
                account_ver.get('total_accounts', 0) > 0
                and len(account_ver.get('verified_accounts', [])) == 0
                and donation_check.get('legitimacy', {}).get('mentioned_organization')
            ):
                # Claimed official org but cannot verify account → force low score
                score = min(score, 35)
            elif donation_check['is_likely_scam']:
                # High risk donation
                score = min(score, max(10, 100 - donation_check['risk_score']))
            elif donation_check.get('red_flags', {}).get('personal_account_only'):
                # Personal account without org - check if has verifiable info
                individual = donation_check.get('individual_legitimacy')
                if individual and individual.get('legitimacy') == 'likely-legitimate':
                    # Has hospital, address, phone → needs review but not scam
                    score = min(score, 80)  # Increased to reach needs-review threshold
                elif individual and individual.get('legitimacy') == 'uncertain':
                    # Some info but not enough
                    score = min(score, 55)
                else:
                    # No verifiable info → high risk
                    score = min(score, 30)
        
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
        donation_check: Optional[Dict] = None,
        temporal_check: Optional[Dict] = None
    ) -> str:
        """
        Determine verdict - ALL posts go through ALL checks
        Any critical red flag = FAKE, regardless of other factors
        """
        
        # === CRITICAL RED FLAGS (AUTO FAIL) ===
        
        # 1. Fake bank account (most critical!)
        if donation_check:
            account_ver = donation_check.get('account_verification', {})
            if account_ver.get('has_fake_accounts'):
                return "donation-scam"  # Fake account = scam, even if news is real
            
            # Unverified account with org mentioned
            if account_ver.get('total_accounts', 0) > 0 and len(account_ver.get('verified_accounts', [])) == 0:
                if donation_check['legitimacy'].get('mentioned_organization'):
                    return "donation-scam"  # Claimed org but wrong account
        
        # 2. External fact-checkers debunked it
        if external_check:
            google = external_check.get('fact_checking', {}).get('google_factcheck', {})
            if google.get('found') and google.get('verdict') == 'false':
                return "likely-false"  # Fact-checkers say false
        
        # 3. Old news manipulation
        if temporal_check and temporal_check.get('is_old_news_manipulation'):
            return "likely-false"  # Old news being used to mislead
        
        # 4. High spam behavior
        if spam_behavior["is_spam"]:
            return "likely-false"  # Being spammed across sites
        
        # 5. High spam language score
        if spam_lang["spam_score"] > 0.5:
            return "likely-false"  # Too much clickbait
        
        # 6. Multiple suspicious facts
        if fact_check and fact_check.get('total_suspicious', 0) >= 2:
            return "likely-false"  # Facts don't add up
        
        # 7. Personal account in donation without org
        if donation_check and donation_check.get('red_flags', {}).get('personal_account_only'):
            # NEW: Check if has verifiable information
            individual = donation_check.get('individual_legitimacy')
            if individual and individual.get('legitimacy') == 'likely-legitimate':
                # Has hospital, address, phone → needs-review (not auto scam)
                pass  # Continue to score-based verdict
            else:
                # No verifiable info → scam
                return "donation-scam"
        
        # 8. High risk donation (other indicators)
        if donation_check and donation_check['is_likely_scam']:
            # Check individual legitimacy first
            individual = donation_check.get('individual_legitimacy')
            if individual and individual.get('legitimacy') == 'likely-legitimate':
                # Has verifiable info → not auto scam
                pass  # Continue to score-based verdict
            else:
                return "donation-scam"
        
        # 9. Unique content with no authority (but skip for donations with news verification)
        if duplication["is_unique"] and not authority["has_authority"]:
            # Exception: donation posts verified by news/gov are OK
            if donation_check and (donation_check['legitimacy'].get('verified_by_news') or donation_check['legitimacy'].get('verified_by_gov')):
                pass  # Don't fail
            else:
                return "likely-false"  # Only 1 source, not trusted
        
        # === NO CRITICAL RED FLAGS - USE TRUST SCORE ===
        
        # External fact-checkers verified it
        if external_check:
            google = external_check.get('fact_checking', {}).get('google_factcheck', {})
            if google.get('found') and google.get('verdict') == 'true':
                return "verified"  # Fact-checkers say true
        
        # Trust score thresholds:
        # >=90: verified (tin thật)
        # 75-89: needs-review (cân nhắc, gửi link)
        # 50-74: likely-false (có thể giả)
        # <50: likely-false (auto giả)
        
        if trust_score >= 90:
            return "verified"
        elif trust_score >= 75:
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
        donation_check: Optional[Dict] = None,
        temporal_check: Optional[Dict] = None
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
        
        # Temporal manipulation
        if temporal_check and temporal_check.get('is_old_news_manipulation'):
            reasons.append(f"Old news manipulation detected: {'; '.join(temporal_check.get('reasons', []))}")
        
        if not reasons:
            reasons.append("Mixed signals, requires manual review")
        
        return " | ".join(reasons)

