import asyncio
from pathlib import Path

import pytest

from data_loader import load_datasets
from trustscore.realtime_verifier import RealtimeVerifier


@pytest.fixture(scope="module")
def datasets():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    return load_datasets(data_dir)


@pytest.mark.asyncio
async def test_clickbait_detection_penalizes_score(datasets):
    engine = RealtimeVerifier(datasets=datasets)
    text = "GIẬT GÂN!!! bạn sẽ không tin điều này"
    result = await engine.verify(text, [], language="vi")
    assert result["trust_score"] < 50
    assert "clickbait" in result["flags"]


@pytest.mark.asyncio
async def test_donation_account_mismatch_triggers_override(datasets):
    engine = RealtimeVerifier(datasets=datasets)
    text = (
        "Kêu gọi ủng hộ gấp cho Vietnam Red Cross, "
        "chuyển khoản STK 0000000000 ngân hàng ABC"
    )
    result = await engine.verify(text, [], language="vi")
    assert result["verdict"] == "donation-scam"
    assert any("donation-scam" in flag for flag in result["flags"])


@pytest.mark.asyncio
async def test_old_news_reposted_sets_flag(datasets):
    engine = RealtimeVerifier(datasets=datasets)
    articles = [
        {
            "domain": "vnexpress.net",
            "first_seen": "2020-01-01T00:00:00Z",
            "published_time": "2020-01-01T00:00:00Z",
        }
    ]
    text = "Tin này vừa xảy ra hôm nay!"
    result = await engine.verify(text, articles, language="vi")
    assert result["verdict"] == "likely-false"
    assert any("old-news" in flag for flag in result["flags"])


@pytest.mark.asyncio
async def test_domain_similarity_flags_typosquatting(datasets):
    engine = RealtimeVerifier(datasets=datasets)
    articles = [{"domain": "reutrers.com", "url": "http://reutrers.com/fake"}]
    result = await engine.verify("Neutral content", articles, language="en")
    assert result["trust_score"] < 60
    assert any("typosquatting" in flag for flag in result["flags"])


@pytest.mark.asyncio
async def test_external_factcheck_false_override(datasets):
    engine = RealtimeVerifier(datasets=datasets)
    text = "Elon Musk bought Facebook for 500 billion USD"
    result = await engine.verify(text, [], language="en")
    assert result["verdict"] == "likely-false"
    assert any("debunked" in flag for flag in result["flags"])
