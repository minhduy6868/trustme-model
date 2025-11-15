"""
Fact Extraction and Verification
Extract specific claims (numbers, dates, names, places) and verify them
"""

import re
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import calendar


class FactExtractor:
    """Extract factual claims from text"""
    
    @staticmethod
    def extract_numbers(text: str) -> List[Dict[str, Any]]:
        """Extract numbers and numeric claims"""
        patterns = [
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(triệu|tỷ|nghìn|million|billion|thousand|%|percent)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(người|cases|deaths|USD|VND|đô|đồng)',
        ]
        
        found_numbers = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                number_str = match.group(1).replace(',', '')
                unit = match.group(2)
                
                try:
                    number = float(number_str)
                    found_numbers.append({
                        'value': number,
                        'unit': unit,
                        'text': match.group(0),
                        'position': match.start()
                    })
                except:
                    pass
        
        return found_numbers
    
    @staticmethod
    def extract_dates(text: str) -> List[Dict[str, Any]]:
        """Extract dates and time references"""
        patterns = [
            # DD/MM/YYYY or DD-MM-YYYY
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 'dmy'),
            # YYYY/MM/DD
            (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', 'ymd'),
            # Month YYYY
            (r'(tháng|month)\s+(\d{1,2})[,\s]+(\d{4})', 'month_year'),
            # Năm YYYY
            (r'(năm|year)\s+(\d{4})', 'year'),
        ]
        
        found_dates = []
        for pattern, date_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                found_dates.append({
                    'text': match.group(0),
                    'type': date_type,
                    'position': match.start(),
                    'groups': match.groups()
                })
        
        return found_dates
    
    @staticmethod
    def extract_names(text: str) -> List[Dict[str, Any]]:
        """Extract person names (simple heuristic)"""
        # Vietnamese names pattern: Title + Name (2-4 words capitalized)
        vn_pattern = r'(Ông|Bà|Anh|Chị|Tiến sĩ|Giáo sư|Bác sĩ|Thủ tướng|Chủ tịch|Tổng thống)\s+([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+(?:\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+){1,3})'
        
        # English names pattern
        en_pattern = r'(Mr\.|Mrs\.|Ms\.|Dr\.|President|Prime Minister|CEO)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})'
        
        found_names = []
        
        for pattern in [vn_pattern, en_pattern]:
            matches = re.finditer(pattern, text)
            for match in matches:
                found_names.append({
                    'title': match.group(1),
                    'name': match.group(2),
                    'full_text': match.group(0),
                    'position': match.start()
                })
        
        return found_names
    
    @staticmethod
    def extract_places(text: str) -> List[Dict[str, Any]]:
        """Extract location mentions"""
        # Common places
        vn_places = [
            'Hà Nội', 'TP HCM', 'Đà Nẵng', 'Hải Phòng', 'Cần Thơ',
            'Việt Nam', 'Hoa Kỳ', 'Mỹ', 'Trung Quốc', 'Nhật Bản'
        ]
        
        en_places = [
            'Vietnam', 'United States', 'USA', 'China', 'Japan',
            'New York', 'Washington', 'London', 'Paris', 'Tokyo'
        ]
        
        all_places = vn_places + en_places
        
        found_places = []
        for place in all_places:
            pattern = r'\b' + re.escape(place) + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                found_places.append({
                    'place': place,
                    'text': match.group(0),
                    'position': match.start()
                })
        
        return found_places


class FactVerifier:
    """Verify extracted facts against common sense and databases"""
    
    @staticmethod
    def verify_numbers(numbers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if numbers are reasonable"""
        suspicious = []
        
        for num in numbers:
            value = num['value']
            unit = num['unit'].lower()
            
            # Check for unreasonable numbers
            if 'tỷ' in unit or 'billion' in unit:
                if value > 10:  # >10 billion of anything is suspicious
                    suspicious.append({
                        'claim': num['text'],
                        'reason': 'Extremely large number (>10 billion)'
                    })
            
            if '%' in unit or 'percent' in unit:
                if value > 100:
                    suspicious.append({
                        'claim': num['text'],
                        'reason': 'Percentage > 100%'
                    })
            
            if 'người' in unit or 'deaths' in unit or 'cases' in unit:
                if value > 1000000000:  # >1 billion people
                    suspicious.append({
                        'claim': num['text'],
                        'reason': 'Unrealistic population number'
                    })
        
        return {
            'total_numbers': len(numbers),
            'suspicious_count': len(suspicious),
            'suspicious_claims': suspicious,
            'is_suspicious': len(suspicious) > 0
        }
    
    @staticmethod
    def verify_dates(dates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if dates are reasonable"""
        suspicious = []
        current_year = datetime.now().year
        
        for date_info in dates:
            date_type = date_info['type']
            groups = date_info['groups']
            
            try:
                if date_type == 'year':
                    year = int(groups[1])
                    if year > current_year:
                        suspicious.append({
                            'claim': date_info['text'],
                            'reason': 'Future date referenced as past event'
                        })
                    elif year < 1900:
                        suspicious.append({
                            'claim': date_info['text'],
                            'reason': 'Very old date (< 1900)'
                        })
                
                elif date_type == 'dmy':
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    if year > current_year:
                        suspicious.append({
                            'claim': date_info['text'],
                            'reason': 'Future date'
                        })
                    elif month > 12 or day > 31:
                        suspicious.append({
                            'claim': date_info['text'],
                            'reason': 'Invalid date format'
                        })
            except:
                pass
        
        return {
            'total_dates': len(dates),
            'suspicious_count': len(suspicious),
            'suspicious_claims': suspicious,
            'is_suspicious': len(suspicious) > 0
        }
    
    @staticmethod
    def verify_names(names: List[Dict[str, Any]], language: str = 'vi') -> Dict[str, Any]:
        """Check name spelling and existence"""
        # Known misspellings
        known_people = {
            'vi': {
                'biden': ['Biden', 'Baiden', 'Bayden'],
                'trump': ['Trump', 'Trum', 'Tromp'],
                'obama': ['Obama', 'Obam', 'Obamma'],
            },
            'en': {
                'biden': ['Biden', 'Baiden'],
                'trump': ['Trump', 'Tromp'],
            }
        }
        
        suspicious = []
        
        for name_info in names:
            name = name_info['name']
            # Simple check: if name has numbers or special chars
            if re.search(r'\d|[!@#$%^&*()]', name):
                suspicious.append({
                    'claim': name_info['full_text'],
                    'reason': 'Name contains invalid characters'
                })
        
        return {
            'total_names': len(names),
            'suspicious_count': len(suspicious),
            'suspicious_claims': suspicious,
            'is_suspicious': len(suspicious) > 0
        }
    
    @staticmethod
    def verify_all(text: str, language: str = 'vi') -> Dict[str, Any]:
        """Extract and verify all facts"""
        extractor = FactExtractor()
        
        # Extract facts
        numbers = extractor.extract_numbers(text)
        dates = extractor.extract_dates(text)
        names = extractor.extract_names(text)
        places = extractor.extract_places(text)
        
        # Verify facts
        numbers_verification = FactVerifier.verify_numbers(numbers)
        dates_verification = FactVerifier.verify_dates(dates)
        names_verification = FactVerifier.verify_names(names, language)
        
        # Calculate fact check score
        total_suspicious = (
            numbers_verification['suspicious_count'] +
            dates_verification['suspicious_count'] +
            names_verification['suspicious_count']
        )
        
        total_facts = len(numbers) + len(dates) + len(names)
        
        if total_facts == 0:
            fact_score = 0.5  # Neutral if no facts found
        else:
            # Lower score if many suspicious facts
            fact_score = max(0, 1.0 - (total_suspicious / total_facts * 2))
        
        return {
            'fact_score': round(fact_score, 3),
            'total_facts': total_facts,
            'total_suspicious': total_suspicious,
            'extracted': {
                'numbers': len(numbers),
                'dates': len(dates),
                'names': len(names),
                'places': len(places)
            },
            'verification': {
                'numbers': numbers_verification,
                'dates': dates_verification,
                'names': names_verification
            }
        }

