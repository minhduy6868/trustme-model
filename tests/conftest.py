"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        "crawler_url": "http://localhost:8000",
        "test_timeout": 30
    }

