"""
Unit tests for storage manager
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_memory_storage():
    """Test in-memory storage"""
    from src.main import StorageManager
    
    storage = StorageManager()
    await storage.connect()
    
    # Test set and get
    await storage.set("test_key", {"value": "test"}, ttl=60)
    result = await storage.get("test_key")
    
    assert result is not None
    assert result["value"] == "test"
    
    # Test exists
    exists = await storage.exists("test_key")
    assert exists is True
    
    # Test delete
    await storage.delete("test_key")
    result = await storage.get("test_key")
    assert result is None


@pytest.mark.asyncio
async def test_redis_storage():
    """Test Redis storage (if available)"""
    try:
        import redis.asyncio as redis
        
        from src.main import StorageManager
        
        storage = StorageManager()
        await storage.connect()
        
        if storage.use_redis:
            # Test set and get
            await storage.set("test_key", {"value": "test"}, ttl=60)
            result = await storage.get("test_key")
            
            assert result is not None
            assert result["value"] == "test"
            
            # Cleanup
            await storage.delete("test_key")
            await storage.close()
        else:
            pytest.skip("Redis not available")
    
    except ImportError:
        pytest.skip("Redis module not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

