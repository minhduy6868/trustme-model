"""
Legitimacy Scorer for Individual Donation Cases
Phân biệt giữa cá nhân thật cần giúp vs scammer
"""

import re
from typing import Dict, Any, List


class LegitimacyScorer:
    """Score legitimacy of individual donation posts"""
    
    # Verifiable information patterns
    HOSPITAL_NAMES = [
        'Bệnh viện', 'Hospital', 'BV', 'Phòng khám',
        'Bạch Mai', 'Chợ Rẫy', 'Nhi Đồng', 'Việt Đức',
        'K Hospital', 'Ung Bướu'
    ]
    
    ADDRESS_PATTERNS = [
        r'\d+\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+',  # Số nhà + tên đường
        r'[qQ]uận\s+\d+',
        r'[pP]hường\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]',
        r'[tT]ỉnh\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]'
    ]
    
    PHONE_PATTERNS = [
        r'0\d{9,10}',  # Vietnamese phone
        r'\+84\s?\d{9,10}'
    ]
    
    @staticmethod
    def extract_verifiable_info(text: str) -> Dict[str, Any]:
        """Extract verifiable information"""
        
        # 1. Hospital names
        hospitals = []
        for hospital in LegitimacyScorer.HOSPITAL_NAMES:
            if hospital.lower() in text.lower():
                hospitals.append(hospital)
        
        # 2. Addresses
        addresses = []
        for pattern in LegitimacyScorer.ADDRESS_PATTERNS:
            matches = re.findall(pattern, text)
            addresses.extend(matches)
        
        # 3. Phone numbers
        phones = []
        for pattern in LegitimacyScorer.PHONE_PATTERNS:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        
        # 4. Names (full Vietnamese names)
        name_pattern = r'(?:bé|em|anh|chị)\s+([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+(?:\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+){1,2})'
        names = re.findall(name_pattern, text)
        
        # 5. Disease/condition names
        conditions = []
        condition_keywords = ['ung thư', 'cancer', 'tai nạn', 'accident', 'bệnh', 'disease', 'mổ', 'surgery']
        for keyword in condition_keywords:
            if keyword in text.lower():
                conditions.append(keyword)
        
        return {
            'hospitals': list(set(hospitals)),
            'addresses': list(set(addresses)),
            'phones': list(set(phones)),
            'names': list(set(names)),
            'conditions': list(set(conditions))
        }
    
    @staticmethod
    def score_legitimacy(
        text: str,
        has_emotional_manipulation: bool,
        has_urgency: bool
    ) -> Dict[str, Any]:
        """
        Score legitimacy of individual donation post
        Phân biệt cá nhân thật vs scammer
        """
        
        verifiable = LegitimacyScorer.extract_verifiable_info(text)
        
        # Calculate verifiability score
        verifiability_score = 0
        
        # Has hospital name (can verify)
        if verifiable['hospitals']:
            verifiability_score += 30
        
        # Has specific address (can verify)
        if verifiable['addresses']:
            verifiability_score += 25
        
        # Has phone number (can contact)
        if verifiable['phones']:
            verifiability_score += 20
        
        # Has full name (can identify)
        if verifiable['names']:
            verifiability_score += 15
        
        # Has medical condition details
        if verifiable['conditions']:
            verifiability_score += 10
        
        # Penalties for scam indicators
        if has_emotional_manipulation:
            verifiability_score -= 25
        
        if has_urgency:
            verifiability_score -= 20
        
        # Determine legitimacy level
        if verifiability_score >= 60:
            legitimacy = "likely-legitimate"
            recommendation = "Has verifiable information. Still recommend checking with hospital/authorities before donating"
        elif verifiability_score >= 30:
            legitimacy = "uncertain"
            recommendation = "Some verifiable information. Strongly recommend verification before donating"
        else:
            legitimacy = "likely-scam"
            recommendation = "Insufficient verifiable information. High risk of scam"
        
        return {
            'verifiability_score': verifiability_score,
            'legitimacy': legitimacy,
            'verifiable_info': verifiable,
            'has_verifiable_info': verifiability_score > 0,
            'recommendation': recommendation
        }

