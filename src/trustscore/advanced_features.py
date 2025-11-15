"""
Advanced verification features
"""

import re
from typing import List, Dict, Any
from datetime import datetime, timedelta
from collections import Counter


class SocialSignalsAnalyzer:
    """Phân tích social signals (engagement patterns)"""
    
    @staticmethod
    def analyze_engagement(article: Dict[str, Any]) -> Dict[str, Any]:
        """Phân tích engagement của bài viết"""
        
        likes = article.get('likes', 0)
        shares = article.get('shares', 0)
        comments = article.get('comments', 0)
        
        # Tính tỷ lệ share/like (bất thường nếu share >> likes)
        if likes > 0:
            share_like_ratio = shares / likes
        else:
            share_like_ratio = 0
        
        # Spam thường có share/like ratio cao (nhiều share, ít like)
        suspicious_ratio = share_like_ratio > 0.5  # >50% shares vs likes
        
        # Bot-driven content thường có engagement cao đột biến
        suspicious_high_engagement = shares > 10000 and comments < 100
        
        return {
            'share_like_ratio': round(share_like_ratio, 3),
            'suspicious_ratio': suspicious_ratio,
            'suspicious_high_engagement': suspicious_high_engagement,
            'total_engagement': likes + shares + comments
        }
    
    @staticmethod
    def check_viral_speed(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Kiểm tra tốc độ lan truyền (viral speed)"""
        
        # Group by publish time
        time_buckets = {}
        
        for article in articles:
            pub_time = article.get('published_time')
            if pub_time:
                try:
                    dt = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                    hour = dt.strftime('%Y-%m-%d %H:00')
                    time_buckets[hour] = time_buckets.get(hour, 0) + 1
                except:
                    pass
        
        if not time_buckets:
            return {'suspicious_viral': False, 'reason': 'No time data'}
        
        max_per_hour = max(time_buckets.values())
        
        # Nếu >10 bài trong 1 giờ = bot-driven
        suspicious_viral = max_per_hour > 10
        
        return {
            'suspicious_viral': suspicious_viral,
            'max_per_hour': max_per_hour,
            'time_distribution': time_buckets
        }


class AccountAnalyzer:
    """Phân tích account behavior"""
    
    @staticmethod
    def analyze_accounts(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phân tích các account đăng bài"""
        
        authors = []
        domains = []
        
        for article in articles:
            author = article.get('author')
            domain = article.get('domain')
            
            if author:
                authors.append(author)
            if domain:
                domains.append(domain)
        
        # Count occurrences
        author_counts = Counter(authors)
        domain_counts = Counter(domains)
        
        # Find suspicious patterns
        suspicious_authors = [author for author, count in author_counts.items() if count > 3]
        suspicious_domains = [domain for domain, count in domain_counts.items() if count > 5]
        
        # Check for bot-like names
        bot_patterns = [
            r'user\d+',
            r'nguoidung\d+',
            r'account\d+',
            r'^[a-z]{8,}$',  # Random string
        ]
        
        likely_bots = []
        for author in authors:
            if author and any(re.match(pattern, author.lower()) for pattern in bot_patterns):
                likely_bots.append(author)
        
        return {
            'total_authors': len(set(authors)),
            'total_domains': len(set(domains)),
            'suspicious_authors': suspicious_authors,
            'suspicious_domains': suspicious_domains,
            'likely_bots': likely_bots,
            'has_suspicious_accounts': len(suspicious_authors) > 0 or len(likely_bots) > 0
        }


class ContentQualityAnalyzer:
    """Phân tích chất lượng nội dung"""
    
    @staticmethod
    def analyze_writing_quality(text: str) -> Dict[str, Any]:
        """Phân tích chất lượng viết"""
        
        # Count errors
        grammar_issues = 0
        
        # Excessive caps
        caps_count = sum(1 for c in text if c.isupper())
        total_letters = sum(1 for c in text if c.isalpha())
        caps_ratio = caps_count / total_letters if total_letters > 0 else 0
        
        if caps_ratio > 0.3:  # >30% caps
            grammar_issues += 1
        
        # Excessive punctuation
        punct_pattern = r'[!?]{3,}'
        excessive_punct = len(re.findall(punct_pattern, text))
        if excessive_punct > 0:
            grammar_issues += 1
        
        # Spelling mistakes (simple check)
        # Repeated characters (vddddddd)
        repeated = len(re.findall(r'(.)\1{3,}', text))
        if repeated > 2:
            grammar_issues += 1
        
        # Missing spaces (lỗi.Không có space)
        missing_spaces = len(re.findall(r'[a-zàáảãạ]\.[A-ZÀÁẢÃẠ]', text))
        if missing_spaces > 2:
            grammar_issues += 1
        
        quality_score = max(0, 1.0 - (grammar_issues * 0.15))
        
        return {
            'quality_score': round(quality_score, 3),
            'grammar_issues': grammar_issues,
            'caps_ratio': round(caps_ratio, 3),
            'excessive_punctuation': excessive_punct,
            'is_low_quality': grammar_issues >= 3
        }
    
    @staticmethod
    def detect_ai_generated(text: str) -> Dict[str, Any]:
        """Phát hiện nội dung do AI tạo (có thể là fake)"""
        
        # AI-generated text patterns (simple heuristics)
        ai_indicators = [
            r'as an AI',
            r'I apologize',
            r'I don\'t have',
            r'based on the information',
            r'it is important to note',
            r'in conclusion',
        ]
        
        ai_pattern_count = sum(1 for pattern in ai_indicators if re.search(pattern, text, re.IGNORECASE))
        
        # Check for overly formal/structured content
        paragraph_count = len(text.split('\n\n'))
        avg_sentence_length = len(text.split('.')) / paragraph_count if paragraph_count > 0 else 0
        
        # AI tends to have very consistent sentence length
        is_overly_structured = avg_sentence_length > 5 and paragraph_count > 3
        
        likely_ai = ai_pattern_count > 0 or is_overly_structured
        
        return {
            'likely_ai_generated': likely_ai,
            'ai_indicator_count': ai_pattern_count,
            'is_overly_structured': is_overly_structured
        }


class CrossReferenceChecker:
    """Kiểm tra cross-references trong bài viết"""
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Trích xuất URLs từ text"""
        url_pattern = r'https?://[^\s]+'
        return re.findall(url_pattern, text)
    
    @staticmethod
    def check_citations(text: str, found_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Kiểm tra trích dẫn"""
        
        cited_urls = CrossReferenceChecker.extract_urls(text)
        
        if not cited_urls:
            return {
                'has_citations': False,
                'citation_count': 0
            }
        
        # Check if cited URLs are from trusted sources
        trusted_citations = 0
        for url in cited_urls:
            url_lower = url.lower()
            if any(trusted in url_lower for trusted in ['reuters', 'bbc', 'vnexpress', 'gov']):
                trusted_citations += 1
        
        return {
            'has_citations': True,
            'citation_count': len(cited_urls),
            'trusted_citations': trusted_citations,
            'citation_quality': trusted_citations / len(cited_urls) if cited_urls else 0
        }


class AdvancedVerifier:
    """Kết hợp tất cả advanced features"""
    
    def __init__(self):
        self.social_analyzer = SocialSignalsAnalyzer()
        self.account_analyzer = AccountAnalyzer()
        self.quality_analyzer = ContentQualityAnalyzer()
        self.crossref_checker = CrossReferenceChecker()
    
    async def analyze_all(self, text: str, found_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Chạy tất cả advanced analysis"""
        
        # Social signals
        social_signals = []
        for article in found_articles:
            if article.get('likes') or article.get('shares'):
                signal = self.social_analyzer.analyze_engagement(article)
                social_signals.append(signal)
        
        viral_check = self.social_analyzer.check_viral_speed(found_articles)
        
        # Account analysis
        account_analysis = self.account_analyzer.analyze_accounts(found_articles)
        
        # Content quality
        quality = self.quality_analyzer.analyze_writing_quality(text)
        ai_check = self.quality_analyzer.detect_ai_generated(text)
        
        # Cross-reference
        citations = self.crossref_checker.check_citations(text, found_articles)
        
        # Calculate advanced score modifier
        score_modifier = 0
        
        # Penalties
        if viral_check.get('suspicious_viral'):
            score_modifier -= 15
        
        if account_analysis.get('has_suspicious_accounts'):
            score_modifier -= 10
        
        if quality.get('is_low_quality'):
            score_modifier -= 10
        
        # Bonuses
        if citations.get('has_citations') and citations.get('citation_quality', 0) > 0.5:
            score_modifier += 10
        
        return {
            'social_signals': {
                'suspicious_engagement': sum(1 for s in social_signals if s.get('suspicious_ratio')),
                'viral_check': viral_check
            },
            'account_analysis': account_analysis,
            'content_quality': quality,
            'ai_detection': ai_check,
            'citations': citations,
            'score_modifier': score_modifier
        }

