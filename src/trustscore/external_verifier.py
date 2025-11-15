"""
External Fact-Checking APIs Integration
Connect to Google Fact Check Tools, Snopes, etc.
"""

import asyncio
from typing import List, Dict, Any, Optional
import httpx
import hashlib


class ExternalFactChecker:
    """Query external fact-checking APIs"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.timeout = 15.0
    
    async def check_google_factcheck(self, claim: str) -> Dict[str, Any]:
        """
        Query Google Fact Check Tools API
        https://toolbox.google.com/factcheck/apis
        """
        api_key = self.api_keys.get('google_factcheck')
        
        if not api_key:
            return {
                'available': False,
                'reason': 'No API key'
            }
        
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            'query': claim[:500],  # Limit query length
            'key': api_key,
            'languageCode': 'vi'
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                
                if response.status_code != 200:
                    return {
                        'available': False,
                        'reason': f'API error: {response.status_code}'
                    }
                
                data = response.json()
                claims = data.get('claims', [])
                
                if not claims:
                    return {
                        'available': True,
                        'found': False,
                        'claim_count': 0
                    }
                
                # Parse claim reviews
                results = []
                for claim_data in claims[:5]:  # Top 5 results
                    reviews = claim_data.get('claimReview', [])
                    for review in reviews:
                        rating = review.get('textualRating', 'Unknown')
                        publisher = review.get('publisher', {}).get('name', 'Unknown')
                        url = review.get('url', '')
                        
                        results.append({
                            'rating': rating,
                            'publisher': publisher,
                            'url': url,
                            'claim': claim_data.get('text', '')
                        })
                
                # Analyze ratings
                false_ratings = ['False', 'Mostly False', 'Fake', 'Incorrect']
                true_ratings = ['True', 'Mostly True', 'Correct', 'Verified']
                
                false_count = sum(1 for r in results if any(fr in r['rating'] for fr in false_ratings))
                true_count = sum(1 for r in results if any(tr in r['rating'] for tr in true_ratings))
                
                return {
                    'available': True,
                    'found': True,
                    'claim_count': len(results),
                    'false_count': false_count,
                    'true_count': true_count,
                    'results': results,
                    'verdict': 'false' if false_count > true_count else 'true' if true_count > 0 else 'mixed'
                }
                
        except Exception as e:
            return {
                'available': False,
                'reason': f'Exception: {str(e)}'
            }
    
    async def check_claimreview_markup(self, urls: List[str]) -> Dict[str, Any]:
        """
        Check if URLs have ClaimReview schema.org markup
        This indicates content has been fact-checked
        """
        has_markup = []
        
        for url in urls[:10]:  # Check first 10 URLs
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url)
                    html = response.text
                    
                    # Look for ClaimReview markup
                    if 'ClaimReview' in html or 'schema.org/ClaimReview' in html:
                        has_markup.append(url)
            except:
                continue
        
        return {
            'total_checked': len(urls[:10]),
            'with_markup': len(has_markup),
            'urls': has_markup
        }
    
    async def check_all(self, claim: str, urls: List[str] = None) -> Dict[str, Any]:
        """Run all external checks"""
        
        # Run checks in parallel
        tasks = [
            self.check_google_factcheck(claim)
        ]
        
        if urls:
            tasks.append(self.check_claimreview_markup(urls))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        google_result = results[0] if not isinstance(results[0], Exception) else {'available': False}
        markup_result = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else {'total_checked': 0}
        
        # Calculate external verification score
        score = 0.5  # Start neutral
        
        if google_result.get('found'):
            if google_result.get('verdict') == 'false':
                score = 0.1  # Very low score if fact-checkers say false
            elif google_result.get('verdict') == 'true':
                score = 0.9  # High score if fact-checkers say true
            else:
                score = 0.5  # Mixed signals
        
        # Boost if ClaimReview markup found
        if markup_result.get('with_markup', 0) > 0:
            score = min(1.0, score + 0.1)
        
        return {
            'external_score': round(score, 3),
            'google_factcheck': google_result,
            'claimreview_markup': markup_result,
            'has_external_verification': google_result.get('found', False) or markup_result.get('with_markup', 0) > 0
        }


class ImageVerificationAPI:
    """Image verification through reverse search"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.timeout = 15.0
    
    async def reverse_image_search(self, image_url: str) -> Dict[str, Any]:
        """
        Reverse image search using Google or TinEye
        Check if image is old or manipulated
        """
        # Note: Google reverse image search doesn't have official API
        # This is a placeholder for future implementation
        
        return {
            'available': False,
            'reason': 'Not implemented yet - need paid API'
        }
    
    async def check_image_metadata(self, image_url: str) -> Dict[str, Any]:
        """
        Check EXIF metadata from image
        Extract date, location, camera info
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(image_url)
                
                if response.status_code != 200:
                    return {'available': False}
                
                # Would need PIL/exifread to extract EXIF
                # Placeholder for now
                
                return {
                    'available': False,
                    'reason': 'EXIF extraction not implemented'
                }
        except:
            return {'available': False}


class CombinedExternalVerifier:
    """Combine all external verification sources"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.fact_checker = ExternalFactChecker(api_keys)
        self.image_verifier = ImageVerificationAPI(api_keys)
    
    async def verify(
        self,
        claim: str,
        urls: List[str] = None,
        images: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run all external verifications
        """
        
        tasks = []
        
        # Fact checking
        tasks.append(self.fact_checker.check_all(claim, urls))
        
        # Image verification (if images provided)
        if images:
            for image_url in images[:3]:  # Check first 3 images
                tasks.append(self.image_verifier.reverse_image_search(image_url))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        fact_check_result = results[0] if not isinstance(results[0], Exception) else {}
        
        return {
            'fact_checking': fact_check_result,
            'external_verification_available': fact_check_result.get('has_external_verification', False)
        }

