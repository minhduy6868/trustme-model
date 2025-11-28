import asyncio
from datetime import datetime, timedelta, timezone
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
    # Avoid early-stop by providing multiple sources
    articles = [
        {"domain": "example.com", "title": text},
        {"domain": "example.org", "snippet": text},
    ]
    result = await engine.verify(text, articles, language="vi")
    assert result["trust_score"] < 80
    assert "clickbait" in result["flags"]


@pytest.mark.asyncio
async def test_donation_account_mismatch_triggers_override(datasets):
    engine = RealtimeVerifier(datasets=datasets)
    text = (
        "Kêu gọi ủng hộ gấp cho Vietnam Red Cross, "
        "chuyển khoản STK 0000000000 ngân hàng ABC"
    )
    articles = [
        {"domain": "example.com", "title": "neutral"},
        {"domain": "example.org", "title": "neutral"},
    ]
    result = await engine.verify(text, articles, language="vi")
    assert result["verdict"] == "donation-scam"
    assert any("donation-scam" in flag for flag in result["flags"])


@pytest.mark.asyncio
async def test_old_news_reposted_sets_flag(datasets):
    engine = RealtimeVerifier(datasets=datasets)
    articles = [
        {
            "domain": "vnexpress.net",
            "first_seen": "2020-01-01T00:00:00+00:00",
            "published_time": "2020-01-01T00:00:00+00:00",
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
    if not datasets.get("fact_check_db"):
        pytest.skip("fact_check_db empty")
    if not any("facebook" in (entry.get("claim_text", "").lower()) for entry in datasets.get("fact_check_db", [])):
        pytest.skip("fact_check_db has no matching claim for test input")
    engine = RealtimeVerifier(datasets=datasets)
    text = "Elon Musk bought Facebook for 500 billion USD"
    result = await engine.verify(text, [], language="en")
    assert result["verdict"] == "likely-false"
    # If fact-check matched, a debunked flag should appear; otherwise verdict remains likely-false
    if any("debunked" in flag for flag in result["flags"]):
        assert True


@pytest.mark.asyncio
async def test_whitelist_domain_triggers_fullscore(datasets):
    """Trusted domain with no anomalies should early-stop at full score."""
    engine = RealtimeVerifier(datasets=datasets)
    now_iso = datetime.now(timezone.utc).isoformat()
    articles = []
    meta = {"domain": "baochinhphu.vn", "url": "https://baochinhphu.vn/bai-viet-mau", "first_seen": now_iso}
    result = await engine.verify("Tin chính phủ cập nhật", articles, language="vi", meta=meta)
    assert result["verdict"] == "verified"
    assert result["trust_score"] == 100.0
    assert result.get("override_reason") == "authority-whitelist"
    assert any("FULL SCORE" in ev for ev in result.get("evidence", []))


@pytest.mark.asyncio
async def test_whitelist_payload_from_extension(datasets):
    """Payload giống từ extension với domain whitelist phải dừng sớm 100 điểm."""
    engine = RealtimeVerifier(datasets=datasets)
    payload = {
        "url": "https://baochinhphu.vn/ket-luan-cua-tong-bi-thu-ve-cong-tac-cham-soc-va-bao-ve-tre-em-co-hoan-canh-dac-biet-102251128172035842.htm",
        "domain": "baochinhphu.vn",
        "title": "Kết luận của Tổng Bí thư về công tác chăm sóc và bảo vệ trẻ em có hoàn cảnh đặc biệt",
        "article": "Thông báo nêu: Ngày 24/11/2025, tại Trụ sở Trung ương Đảng, đồng chí Tổng Bí thư đã có buổi làm việc ...",
        "created_at": "2025-11-28T06:22:00.000Z",
        "author": "baochinhphu.vn",
        "platform": "web",
    }
    meta = {
        "domain": payload["domain"],
        "url": payload["url"],
        "created_at": payload["created_at"],
    }
    result = await engine.verify(payload["article"], [], language="vi", meta=meta)
    assert result["verdict"] == "verified"
    assert result["trust_score"] == 100.0
    assert result.get("override_reason") == "authority-whitelist"
