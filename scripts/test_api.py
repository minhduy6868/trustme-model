"""
Quick API test script
"""

import httpx
import asyncio


async def test_api():
    """Test Model API endpoints"""
    
    base_url = "http://localhost:8001"
    
    print("Testing Model API...")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test health
        print("\n1. Health check...")
        response = await client.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test metrics
        print("\n2. Metrics...")
        response = await client.get(f"{base_url}/metrics")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test verify
        print("\n3. Verify endpoint...")
        response = await client.post(
            f"{base_url}/verify",
            json={
                "text": "Test claim for verification",
                "language": "en",
                "deep_analysis": False
            }
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Job ID: {data.get('job_id')}")
        
        # Poll result
        if data.get('job_id'):
            job_id = data['job_id']
            print(f"\n4. Polling result for {job_id}...")
            
            for i in range(30):
                await asyncio.sleep(2)
                response = await client.get(f"{base_url}/result/{job_id}")
                result = response.json()
                
                if result['status'] == 'completed':
                    print(f"Completed! Trust score: {result.get('trust_score')}")
                    break
                elif result['status'] == 'failed':
                    print(f"Failed: {result.get('error')}")
                    break
                else:
                    print(f"  [{i+1}/30] Status: {result['status']}")
    
    print("\n" + "=" * 50)
    print("Test completed")


if __name__ == "__main__":
    asyncio.run(test_api())

