"""
Temporal Verification - Detect old news being reposted for manipulation
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


class TemporalVerifier:
    """Verify timeline and detect old news manipulation"""
    
    @staticmethod
    def extract_dates_from_text(text: str) -> List[Dict[str, Any]]:
        """Extract dates mentioned in text"""
        patterns = [
            # DD/MM/YYYY
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 'dmy'),
            # YYYY/MM/DD
            (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', 'ymd'),
            # Tháng X năm YYYY
            (r'tháng\s+(\d{1,2})[,\s]+năm\s+(\d{4})', 'month_year_vn'),
            # Month YYYY
            (r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', 'month_year_en'),
            # Năm YYYY
            (r'năm\s+(\d{4})', 'year_vn'),
            (r'year\s+(\d{4})', 'year_en'),
        ]
        
        found_dates = []
        for pattern, date_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                found_dates.append({
                    'text': match.group(0),
                    'type': date_type,
                    'groups': match.groups(),
                    'position': match.start()
                })
        
        return found_dates
    
    @staticmethod
    def parse_date(date_info: Dict[str, Any]) -> Optional[datetime]:
        """Parse date from extracted info"""
        from datetime import timezone
        date_type = date_info['type']
        groups = date_info['groups']
        
        try:
            if date_type == 'dmy':
                day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                return datetime(year, month, day, tzinfo=timezone.utc)
            
            elif date_type == 'ymd':
                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                return datetime(year, month, day, tzinfo=timezone.utc)
            
            elif date_type in ['month_year_vn', 'month_year_en']:
                month, year = int(groups[0]), int(groups[1])
                return datetime(year, month, 1, tzinfo=timezone.utc)
            
            elif date_type in ['year_vn', 'year_en']:
                year = int(groups[0])
                return datetime(year, 1, 1, tzinfo=timezone.utc)
        
        except:
            pass
        
        return None
    
    @staticmethod
    def check_article_dates(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check publication dates of articles"""
        from datetime import timezone
        now = datetime.now(timezone.utc)
        
        dates = []
        for article in articles:
            pub_time = article.get('published_time')
            if pub_time:
                try:
                    dt = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                    dates.append(dt)
                except:
                    pass
        
        if not dates:
            return {
                'has_dates': False,
                'oldest_date': None,
                'newest_date': None,
                'is_old_news': False
            }
        
        oldest = min(dates)
        newest = max(dates)
        
        # Check if oldest article is very old (>1 year)
        age_days = (now - oldest).days
        is_old_news = age_days > 365
        
        # Check if there's a big gap (old news being reposted)
        date_gap_days = (newest - oldest).days if len(dates) > 1 else 0
        suspicious_gap = date_gap_days > 730  # >2 years gap
        
        return {
            'has_dates': True,
            'oldest_date': oldest.isoformat(),
            'newest_date': newest.isoformat(),
            'age_days': age_days,
            'is_old_news': is_old_news,
            'date_gap_days': date_gap_days,
            'suspicious_gap': suspicious_gap
        }
    
    @staticmethod
    def detect_old_news_manipulation(
        text: str,
        articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect if old news is being reposted for manipulation"""
        
        # Extract dates from text
        text_dates = TemporalVerifier.extract_dates_from_text(text)
        
        # Parse dates
        parsed_dates = []
        for date_info in text_dates:
            dt = TemporalVerifier.parse_date(date_info)
            if dt:
                parsed_dates.append(dt)
        
        # Check article publication dates
        article_dates = TemporalVerifier.check_article_dates(articles)
        
        from datetime import timezone
        now = datetime.now(timezone.utc)
        
        # Detect manipulation patterns
        is_manipulation = False
        manipulation_reasons = []
        
        # Pattern 1: Text mentions recent events ("hôm nay", "today") but articles are old
        if parsed_dates:
            recent_text_dates = [d for d in parsed_dates if (now - d).days < 30]
            if recent_text_dates and article_dates.get('age_days', 0) > 365:
                is_manipulation = True
                manipulation_reasons.append("Text mentions recent dates but source articles are >1 year old")
        
        # Pattern 1b: Text says "today" or "hôm nay" but articles are old
        today_keywords = ['hôm nay', 'today', 'ngày nay', 'hiện nay']
        has_today_keyword = any(keyword in text.lower() for keyword in today_keywords)
        
        if has_today_keyword and article_dates.get('age_days', 0) > 180:  # >6 months
            is_manipulation = True
            manipulation_reasons.append(f"Text says 'today' but articles are {article_dates.get('age_days')} days old")
        
        # Pattern 2: Suspicious date gap in articles
        if article_dates.get('suspicious_gap'):
            is_manipulation = True
            manipulation_reasons.append(f"Articles span {article_dates['date_gap_days']} days (old news being reposted)")
        
        # Pattern 3: Multiple old sources suddenly appearing
        if article_dates.get('is_old_news') and len(articles) > 5:
            # Many old articles appearing together = likely manipulation
            is_manipulation = True
            manipulation_reasons.append("Multiple old articles (>1 year) appearing simultaneously")
        
        # Calculate manipulation score
        manipulation_score = 0
        if is_manipulation:
            manipulation_score = min(100, len(manipulation_reasons) * 30)
        
        return {
            'is_old_news_manipulation': is_manipulation,
            'manipulation_score': manipulation_score,
            'reasons': manipulation_reasons,
            'text_dates_found': len(text_dates),
            'oldest_article_age_days': article_dates.get('age_days'),
            'article_date_check': article_dates
        }

