"""
Donation/Charity Post Detector
Detect fake donation/charity scam posts
"""

import re
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from .llm_adapter import LLMAdapter, LLMFallbackError


def load_official_accounts() -> Dict[str, Any]:
    """Load official bank accounts database"""
    data_dir = os.getenv("DATA_DIR") or Path(__file__).resolve().parents[2] / "data"
    candidates = [
        Path(data_dir) / "official_accounts.json",
        Path(__file__).parent.parent.parent / "config" / "official_accounts.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "organizations" not in data:
                    data = {"organizations": data}
                return data
    return {"organizations": {}}


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
        """Extract bank account numbers and bank names"""
        # Vietnamese bank account patterns
        patterns = [
            (r'(?:STK|số tài khoản|account|acc|TK)[\s:]*(\d{8,16})', None),
            (r'(MB|MBBank)[\s:]*(\d{8,16})', 'MBBank'),
            (r'(VCB|Vietcombank)[\s:]*(\d{8,16})', 'Vietcombank'),
            (r'(TCB|Techcombank)[\s:]*(\d{8,16})', 'Techcombank'),
            (r'(ACB)[\s:]*(\d{8,16})', 'ACB'),
            (r'(VPB|VPBank)[\s:]*(\d{8,16})', 'VPBank'),
            (r'(TPB|TPBank)[\s:]*(\d{8,16})', 'TPBank'),
        ]
        
        accounts = []
        for pattern, bank_name in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get account number (last group)
                account_num = match.groups()[-1]
                accounts.append({
                    'number': account_num,
                    'bank': bank_name or 'Unknown',
                    'context': match.group(0)
                })
        
        return accounts
    
    @staticmethod
    def verify_account_legitimacy(
        extracted_accounts: List[Dict[str, Any]],
        organization_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify if bank accounts are legitimate"""
        
        if not extracted_accounts:
            return {
                'has_accounts': False,
                'verified_accounts': [],
                'fake_accounts': [],
                'warning': None
            }
        
        official_data = load_official_accounts()
        organizations = official_data.get('organizations', {})
        
        verified_accounts = []
        fake_accounts = []
        
        # If organization name is mentioned, check against their official accounts
        if organization_name:
            org_key = organization_name.lower().replace(' ', '_')
            
            for org_id, org_data in organizations.items():
                org_name_lower = org_data['name'].lower()
                
                # Check if this organization is mentioned
                if org_name_lower in organization_name.lower() or org_id in org_key:
                    official_accounts = org_data.get('official_accounts', [])
                    
                    # Compare extracted accounts with official ones
                    for extracted in extracted_accounts:
                        is_verified = False
                        
                        for official in official_accounts:
                            if extracted['number'] == official['account_number']:
                                verified_accounts.append({
                                    'account': extracted,
                                    'organization': org_data['name']
                                })
                                is_verified = True
                                break
                        
                        if not is_verified:
                            # Account không khớp với official account
                            fake_accounts.append({
                                'account': extracted,
                                'expected_org': org_data['name'],
                                'official_accounts': official_accounts
                            })
        
        # Determine warning level
        warning = None
        if fake_accounts and organization_name:
            warning = f"FAKE ACCOUNT DETECTED! Post mentions {organization_name} but uses different account numbers"
        elif not verified_accounts and organization_name:
            warning = f"UNVERIFIED ACCOUNT! Cannot confirm account belongs to {organization_name}"
        elif not organization_name and extracted_accounts:
            warning = "Personal account found. No official organization mentioned"
        
        return {
            'has_accounts': True,
            'total_accounts': len(extracted_accounts),
            'verified_accounts': verified_accounts,
            'fake_accounts': fake_accounts,
            'has_fake_accounts': len(fake_accounts) > 0,
            'warning': warning
        }
    
    @staticmethod
    def check_legitimacy(
        text: str,
        found_articles: List[Dict[str, Any]],
        language: str = 'vi',
        llm: Optional[LLMAdapter] = None,
    ) -> Dict[str, Any]:
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
        
        # NEW: Check legitimacy for individual cases
        try:
            from .legitimacy_scorer import LegitimacyScorer
            legitimacy_check = LegitimacyScorer.score_legitimacy(
                text, has_emotional_manipulation, has_urgency
            )
        except ImportError:
            legitimacy_check = None
        
        # 3. Check legitimate organization mentioned
        mentioned_org = None
        for org in DonationDetector.LEGITIMATE_ORGS:
            if org.lower() in text.lower():
                mentioned_org = org
                has_legitimate_org = True
                break
        else:
            has_legitimate_org = False
        
        # 4. Extract bank accounts
        bank_accounts = DonationDetector.extract_bank_accounts(text)
        
        # 5. NEW: Verify bank accounts against official database
        account_verification = DonationDetector.verify_account_legitimacy(
            bank_accounts,
            mentioned_org
        )
        
        has_personal_account = len(bank_accounts) > 0 and not has_legitimate_org
        has_fake_account = account_verification.get('has_fake_accounts', False)

        llm_intent = None
        llm_label = None
        if llm:
            try:
                llm_label = llm.classify(text, ["scam", "spam", "legit"])
                if llm_label == "scam":
                    llm_intent = "scam"
                elif llm_label == "spam":
                    llm_intent = "spam"
                else:
                    llm_intent = "legit"
            except LLMFallbackError:
                llm_intent = None
        
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
        
        # CRITICAL: Fake account detected (highest priority)
        if has_fake_account:
            risk_score += 90  # Near maximum risk!
        
        if has_emotional_manipulation:
            risk_score += 30
        
        if has_urgency:
            risk_score += 25
        
        if has_personal_account and not has_legitimate_org:
            # NEW: Check if individual case has verifiable info
            if legitimacy_check and legitimacy_check.get('legitimacy') == 'likely-legitimate':
                risk_score += 15  # Lower risk if has verifiable info
            else:
                risk_score += 35  # High risk if no verifiable info
        
        if not verified_by_news and not verified_by_gov:
            risk_score += 20
        
        # Reduce risk if legitimate
        if has_legitimate_org and not has_fake_account:
            risk_score = max(0, risk_score - 30)
        
        if verified_by_news and not has_fake_account:
            risk_score = max(0, risk_score - 20)
        
        if verified_by_gov and not has_fake_account:
            risk_score = max(0, risk_score - 40)

        if llm_intent == "scam":
            risk_score += 25
        elif llm_intent == "spam":
            risk_score += 15
        
        # NEW: Reduce risk if has verifiable information (individual case)
        if legitimacy_check and legitimacy_check.get('has_verifiable_info'):
            verif_score = legitimacy_check.get('verifiability_score', 0)
            if verif_score >= 60:
                risk_score = max(0, risk_score - 30)  # Has hospital, address, phone
            elif verif_score >= 30:
                risk_score = max(0, risk_score - 15)  # Has some info
        
        # Determine if scam
        # If fake account detected, always high risk
        # If personal account but has verifiable info → NOT scam, just needs review
        if has_fake_account:
            is_likely_scam = True
        elif legitimacy_check and legitimacy_check.get('legitimacy') == 'likely-legitimate':
            is_likely_scam = False  # Has verifiable info → not scam
        else:
            is_likely_scam = risk_score >= 60
        
        return {
            'is_donation_post': True,
            'risk_score': min(100, risk_score),
            'is_likely_scam': is_likely_scam,
            'red_flags': {
                'emotional_manipulation': has_emotional_manipulation,
                'urgency': has_urgency,
                'personal_account_only': has_personal_account,
                'no_verification': not verified_by_news and not verified_by_gov,
                'fake_account_detected': has_fake_account,
                'llm_intent': llm_intent
            },
            'legitimacy': {
                'has_legitimate_org': has_legitimate_org,
                'mentioned_organization': mentioned_org,
                'verified_by_news': verified_by_news,
                'verified_by_gov': verified_by_gov,
                'llm_label': llm_label
            },
            'account_verification': account_verification,
            'individual_legitimacy': legitimacy_check,  # NEW: For individual cases
            'bank_accounts_found': len(bank_accounts),
            'warning': 'CRITICAL: Fake bank account detected!' if has_fake_account else ('High risk of donation scam' if is_likely_scam else 'Check carefully before donating')
        }
    
    @staticmethod
    def generate_donation_warning(legitimacy_check: Dict[str, Any]) -> str:
        """Generate warning message for donation posts"""
        
        if not legitimacy_check['is_donation_post']:
            return ""
        
        risk_score = legitimacy_check['risk_score']
        red_flags = legitimacy_check['red_flags']
        legitimacy = legitimacy_check['legitimacy']
        account_verification = legitimacy_check.get('account_verification', {})
        
        warnings = []
        
        # CRITICAL: Fake account (highest priority)
        if red_flags.get('fake_account_detected'):
            fake_accounts = account_verification.get('fake_accounts', [])
            if fake_accounts:
                warnings.append(f"CRITICAL: FAKE BANK ACCOUNT! Post mentions '{legitimacy.get('mentioned_organization')}' but uses different account")
                warnings.append(f"Suspicious account: {fake_accounts[0]['account']['number']}")
        
        # High risk warning
        elif legitimacy_check['is_likely_scam']:
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
        
        # Positive signals (only if no fake account)
        if not red_flags.get('fake_account_detected'):
            if legitimacy['verified_by_gov']:
                warnings.append("SAFE: Verified by government source")
            elif legitimacy['verified_by_news']:
                warnings.append("SAFER: Reported by trusted news")
            elif legitimacy['has_legitimate_org']:
                if account_verification.get('verified_accounts'):
                    warnings.append("VERIFIED: Official organization account confirmed")
                else:
                    warnings.append("Organization name mentioned (verify account before donating)")
            else:
                # Individual case without org
                individual = legitimacy_check.get('individual_legitimacy')
                if individual:
                    if individual.get('legitimacy') == 'likely-legitimate':
                        warnings.append(f"INDIVIDUAL CASE: Has verifiable information (hospital, address, phone). {individual.get('recommendation', '')}")
                    elif individual.get('legitimacy') == 'uncertain':
                        warnings.append(f"INDIVIDUAL CASE: Limited verifiable information. {individual.get('recommendation', '')}")
                    else:
                        warnings.append("INDIVIDUAL CASE: Insufficient information to verify")
        
        if not warnings:
            warnings.append("Exercise caution before donating")
        
        return " | ".join(warnings)
