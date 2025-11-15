"""
Donation/Charity Post Detector
Detect fake donation/charity scam posts
"""

import re
from typing import Dict, Any, List


class DonationDetector:
    """Detect and verify donation/charity posts"""
    
    # Donation keywords
    DONATION_KEYWORDS_VI = [
        'quyên góp', 'kêu gọi', 'ủng hộ', 'giúp đỡ', 'từ thiện',
        'hoàn cảnh khó khăn', 'mổ', 'phẫu thuật', 'bệnh viện',
        'tiền viện phí', 'chữa trị', 'điều trị', 'số tài khoản',
        'chuyển khoản', 'donate', 'charity'
    ]
    
    DONATION_KEYWORDS_EN = [
        'donate', 'donation', 'charity', 'fundraising', 'help',
        'support', 'gofundme', 'account number', 'transfer',
        'contribute', 'aid'
    ]
    
    # Emotional manipulation (red flag)
    EMOTIONAL_MANIPULATION_VI = [
        'thương tâm', 'đáng thương', 'tội nghiệp', 'rơi nước mắt',
        'xót xa', 'đau lòng', 'nghẹn ngào', 'khóc ròng',
        'cầu xin', 'van xin', 'nước mắt', 'cơ cực',
        'khổ sở', 'đói khát', 'lâm cảnh', 'bế tắc'
    ]
    
    # Urgency (red flag)
    URGENCY_WORDS = [
        'gấp', 'khẩn cấp', 'cấp bách', 'ngay', 'nhanh tay',
        'còn 1 ngày', 'sắp không kịp', 'urgent', 'emergency'
    ]
    
    # Legitimate organizations
    LEGITIMATE_ORGS = [
        'Mặt trận Tổ quốc', 'Hội Chữ thập đỏ', 'UNICEF',
        'Red Cross', 'Quỹ tấm lòng Việt', 'Báo Thanh Niên',
        'VTV', 'Ủy ban MTTQ', 'chính quyền', 'UBND'
    ]
    
    @staticmethod
    def is_donation_post(text: str, language: str = 'vi') -> bool:
        """Check if this is a donation/charity post"""
        keywords = DonationDetector.DONATION_KEYWORDS_VI if language == 'vi' else DonationDetector.DONATION_KEYWORDS_EN
        
        count = sum(1 for keyword in keywords if keyword in text.lower())
        return count >= 2  # At least 2 donation keywords
    
    @staticmethod
    def extract_bank_accounts(text: str) -> List[Dict[str, Any]]:
        """Extract bank account numbers"""
        # Vietnamese bank account patterns
        patterns = [
            r'(?:STK|số tài khoản|account|acc|TK)[\s:]*(\d{8,16})',
            r'(?:MB|VCB|TCB|ACB|VPB|TPB|Techcombank|Vietcombank|MBBank)[\s:]*(\d{8,16})',
        ]
        
        accounts = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                accounts.append({
                    'number': match.group(1),
                    'context': match.group(0)
                })
        
        return accounts
    
    @staticmethod
    def check_legitimacy(text: str, found_articles: List[Dict[str, Any]], language: str = 'vi') -> Dict[str, Any]:
        """Check if donation post is legitimate"""
        
        # 1. Check emotional manipulation
        emotional_count = sum(
            1 for word in DonationDetector.EMOTIONAL_MANIPULATION_VI 
            if word in text.lower()
        )
        has_emotional_manipulation = emotional_count >= 3
        
        # 2. Check urgency
        urgency_count = sum(
            1 for word in DonationDetector.URGENCY_WORDS 
            if word in text.lower()
        )
        has_urgency = urgency_count >= 2
        
        # 3. Check legitimate organization
        has_legitimate_org = any(
            org.lower() in text.lower() 
            for org in DonationDetector.LEGITIMATE_ORGS
        )
        
        # 4. Extract bank accounts
        bank_accounts = DonationDetector.extract_bank_accounts(text)
        has_personal_account = len(bank_accounts) > 0 and not has_legitimate_org
        
        # 5. Check if verified by news
        verified_by_news = False
        for article in found_articles:
            domain = article.get('domain', '').lower()
            if any(news in domain for news in ['vnexpress', 'tuoitre', 'thanhnien', 'vtv', 'vov']):
                verified_by_news = True
                break
        
        # 6. Check if has official government verification
        verified_by_gov = False
        for article in found_articles:
            domain = article.get('domain', '').lower()
            if 'gov' in domain or 'chinhphu' in domain:
                verified_by_gov = True
                break
        
        # Calculate risk score
        risk_score = 0
        
        if has_emotional_manipulation:
            risk_score += 30
        
        if has_urgency:
            risk_score += 25
        
        if has_personal_account and not has_legitimate_org:
            risk_score += 35  # Personal account without org = very risky
        
        if not verified_by_news and not verified_by_gov:
            risk_score += 20
        
        # Reduce risk if legitimate
        if has_legitimate_org:
            risk_score = max(0, risk_score - 30)
        
        if verified_by_news:
            risk_score = max(0, risk_score - 20)
        
        if verified_by_gov:
            risk_score = max(0, risk_score - 40)
        
        # Determine if scam
        is_likely_scam = risk_score >= 60
        
        return {
            'is_donation_post': True,
            'risk_score': min(100, risk_score),
            'is_likely_scam': is_likely_scam,
            'red_flags': {
                'emotional_manipulation': has_emotional_manipulation,
                'urgency': has_urgency,
                'personal_account_only': has_personal_account,
                'no_verification': not verified_by_news and not verified_by_gov
            },
            'legitimacy': {
                'has_legitimate_org': has_legitimate_org,
                'verified_by_news': verified_by_news,
                'verified_by_gov': verified_by_gov
            },
            'bank_accounts_found': len(bank_accounts),
            'warning': 'High risk of donation scam' if is_likely_scam else 'Check carefully before donating'
        }
    
    @staticmethod
    def generate_donation_warning(legitimacy_check: Dict[str, Any]) -> str:
        """Generate warning message for donation posts"""
        
        if not legitimacy_check['is_donation_post']:
            return ""
        
        risk_score = legitimacy_check['risk_score']
        red_flags = legitimacy_check['red_flags']
        legitimacy = legitimacy_check['legitimacy']
        
        warnings = []
        
        # High risk warning
        if legitimacy_check['is_likely_scam']:
            warnings.append("HIGH RISK: Likely donation scam")
        
        # Specific warnings
        if red_flags['personal_account_only']:
            warnings.append("Only personal bank account provided (no official organization)")
        
        if red_flags['emotional_manipulation']:
            warnings.append("Uses excessive emotional language")
        
        if red_flags['urgency']:
            warnings.append("Creates false sense of urgency")
        
        if red_flags['no_verification']:
            warnings.append("Not verified by any news or government source")
        
        # Positive signals
        if legitimacy['verified_by_gov']:
            warnings.append("SAFE: Verified by government source")
        elif legitimacy['verified_by_news']:
            warnings.append("SAFER: Reported by trusted news")
        elif legitimacy['has_legitimate_org']:
            warnings.append("Has legitimate organization name")
        
        if not warnings:
            warnings.append("Exercise caution before donating")
        
        return " | ".join(warnings)

