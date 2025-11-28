import asyncio
from typing import Dict, Any

import httpx
import pytest

import main


WHITELIST_PAYLOAD: Dict[str, Any] = {
    "url": "https://baochinhphu.vn/ket-luan-cua-tong-bi-thu-ve-cong-tac-cham-soc-va-bao-ve-tre-em-co-hoan-canh-dac-biet-102251128172035842.htm",
    "title": "Kết luận của Tổng Bí thư về công tác chăm sóc và bảo vệ trẻ em có hoàn cảnh đặc biệt",
    "domain": "baochinhphu.vn",
    "article": (
        "Hai bé sinh đôi bị bỏ rơi tại TP. Bến Cát, Bình Dương - Ảnh: TTXVN\n"
        "Thông báo nêu: Ngày 24/11/2025, tại Trụ sở Trung ương Đảng, đồng chí Tổng Bí thư đã có buổi làm việc "
        "với Đảng ủy Bộ Y tế, các ban, bộ, ngành Trung ương và địa phương (gồm các đồng chí Uỷ viên Trung ương "
        "Đảng: Phạm Gia Túc, Chánh Văn phòng Trung ương Đảng; Nguyễn Đắc Vinh, Chủ nhiệm Uỷ ban Văn hoá-Xã hội "
        "của Quốc hội; Đào Hồng Lan, Bộ trưởng Bộ Y tế; Lâm Thị Phương Thanh, Phó Chánh Văn phòng Thường trực; "
        "Đảng uỷ các Bộ: Y tế, Giáo dục và Đào tạo, Tài chính, Nội vụ; Đảng uỷ Mặt trận Tổ quốc, các đoàn thể "
        "Trung ương; Thành uỷ Hà Nội; Hội Bảo vệ quyền trẻ em Việt Nam; Văn phòng Trung ương Đảng) về công tác "
        "chăm sóc và bảo vệ trẻ em có hoàn cảnh đặc biệt khó khăn. Sau khi nghe Đảng uỷ Bộ Y tế báo cáo, ý kiến "
        "của các cơ quan liên quan, đồng chí Tổng Bí thư kết luận như sau:\n"
        "Trẻ em có hoàn cảnh đặc biệt là những người chịu nhiều thiệt thòi, nhất là về điều kiện sống, dinh dưỡng, "
        "giáo dục và chăm sóc sức khoẻ. Chăm lo cho trẻ em có hoàn cảnh đặc biệt không chỉ để chữa lành những bất "
        "hạnh hiện tại mà chính là sự chăm lo cho tương lai và sự phát triển bền vững đất nước, thể hiện tính nhân "
        "văn, ưu việt của chế độ xã hội chủ nghĩa..."
    ),
    "created_at": "2025-11-28T06:22:00.000Z",
    "author": "baochinhphu.vn",
    "image_urls": [
        "https://bcp.cdnchinhphu.vn/zoom/600_315/334894974524682240/2025/11/28/avatar1764324963387-1764324963783899671790.jpg"
    ],
}


@pytest.mark.asyncio
async def test_whitelist_domain_short_circuits_crawler():
    async with main.lifespan(main.app):
        # Đảm bảo domain nằm trong bộ whitelist dữ liệu sau khi app khởi động
        whitelist = [d.lower() for d in (main.DATASETS.get("domain_whitelist") or [])]
        if whitelist and "baochinhphu.vn" not in whitelist:
            pytest.skip("Dataset không chứa baochinhphu.vn; bỏ qua kiểm tra whitelist.")

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            domain = WHITELIST_PAYLOAD["domain"]
            meta = {
                "domain_frequency": {domain: 1},
                "domain": domain,
                "image_urls": WHITELIST_PAYLOAD["image_urls"],
            }
            found_articles = [
                {
                    "domain": domain,
                    "url": WHITELIST_PAYLOAD["url"],
                    "first_seen": WHITELIST_PAYLOAD["created_at"],
                    "published_time": WHITELIST_PAYLOAD["created_at"],
                }
            ]

            resp = await client.post(
                "/verify",
                json={
                    "text": WHITELIST_PAYLOAD["article"],
                    "url": WHITELIST_PAYLOAD["url"],
                    "language": "vi",
                    "deep_analysis": True,
                    # Payload giống extension gửi lên
                    "found_articles": found_articles,
                    "meta": meta,
                    "author": WHITELIST_PAYLOAD["author"],
                    "image_urls": WHITELIST_PAYLOAD["image_urls"],
                },
            )
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            # Poll kết quả để chắc chắn background task đã chạy và rẽ nhánh whitelist
            for _ in range(30):
                result_resp = await client.get(f"/result/{job_id}")
                result_resp.raise_for_status()
                result = result_resp.json()
                if result["status"] == "completed":
                    assert result.get("override_reason") == "authority-whitelist"
                    assert result.get("trust_score") in (100, 100.0)
                    crawl_stats = result.get("crawl_stats") or {}
                    assert crawl_stats.get("skipped") is True
                    assert crawl_stats.get("reason") == "domain_whitelist"
                    assert crawl_stats.get("domain") == domain
                    assert result.get("verdict") == "verified"
                    components = result.get("components") or {}
                    assert components  # có component trả về
                    return
                await asyncio.sleep(0.2)

            raise AssertionError("Job không hoàn tất sau khi gửi whitelist payload")


@pytest.mark.asyncio
async def test_facebook_trusted_page_short_circuits_crawler():
    async with main.lifespan(main.app):
        trusted_pages = [s for s in (main.DATASETS.get("trusted_sources") or []) if s.get("type") == "page"]
        page_hit = next((p for p in trusted_pages if "facebook.com/thongtinchinhphu" in (p.get("identifier") or "")), None)
        if not page_hit:
            pytest.skip("Dataset không chứa trusted page 'thongtinchinhphu'; bỏ qua kiểm tra.")

        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            url = "https://www.facebook.com/thongtinchinhphu/posts/123456789"
            author = "Thông tin Chính phủ"
            domain = "facebook.com"
            meta = {
                "domain_frequency": {domain: 1},
                "domain": domain,
                "image_urls": [],
            }
            found_articles = [
                {
                    "domain": domain,
                    "url": url,
                    "first_seen": None,
                    "published_time": None,
                }
            ]

            resp = await client.post(
                "/verify",
                json={
                    "text": "Bản tin cập nhật chính thức từ Thông tin Chính phủ.",
                    "url": url,
                    "language": "vi",
                    "deep_analysis": True,
                    "found_articles": found_articles,
                    "meta": meta,
                    "author": author,
                },
            )
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            for _ in range(30):
                result_resp = await client.get(f"/result/{job_id}")
                result_resp.raise_for_status()
                result = result_resp.json()
                if result["status"] == "completed":
                    assert result.get("override_reason") == "authority-whitelist"
                    assert result.get("trust_score") in (100, 100.0)
                    crawl_stats = result.get("crawl_stats") or {}
                    assert crawl_stats.get("skipped") is True
                    assert crawl_stats.get("reason") == "trusted_page"
                    assert crawl_stats.get("domain") == domain
                    assert result.get("verdict") == "verified"
                    return
                await asyncio.sleep(0.2)

            raise AssertionError("Job không hoàn tất sau khi gửi trusted Facebook payload")
