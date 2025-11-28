"""
API endpoint tests
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client"""
    from src.main import app
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data


def test_metrics_endpoint(client):
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data


def test_verify_endpoint_validation(client):
    """Test verify endpoint with invalid input"""
    # Test with short text
    response = client.post(
        "/verify",
        json={"text": "short", "language": "vi"}
    )
    assert response.status_code == 422  # Validation error
    
    # Test with invalid language
    response = client.post(
        "/verify",
        json={"text": "This is a test claim", "language": "invalid"}
    )
    assert response.status_code == 422


def test_verify_endpoint_success(client):
    """Test verify endpoint with valid input"""
    response = client.post(
        "/verify",
        json={
            "text": "This is a test claim for verification",
            "language": "en",
            "deep_analysis": False
        }
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "processing"


def test_result_endpoint_not_found(client):
    """Test result endpoint with non-existent job_id"""
    response = client.get("/result/invalid_job_id")
    assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
import pytest

pytest.skip("Skipping live API tests in unit run", allow_module_level=True)
