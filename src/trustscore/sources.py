from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import datetime
from urllib.parse import urlparse, quote_plus

import httpx
from duckduckgo_search import DDGS
from email.utils import parsedate_to_datetime
import trafilatura
import feedparser

from .config import get_settings
from .models import DocumentPayload, SourceEvidence


class SourceAggregator:
    def __init__(self, *, timeout: float = 10.0) -> None:
        self._settings = get_settings()
        self._timeout = timeout

    async def gather(
        self,
        query: str,
        url: str | None,
        *,
        provided_documents: Sequence[DocumentPayload] | None = None,
    ) -> list[SourceEvidence]:
        tasks = [
            self._query_fact_check_api(query),
            self._query_news_api(query),
            self._query_search_api(query, url),
            self._query_duckduckgo_news(query),
            self._query_google_rss(query),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        evidence: list[SourceEvidence] = []
        for result in results:
            if isinstance(result, list):
                evidence.extend(result)
        if provided_documents:
            evidence.extend(self._from_provided_documents(provided_documents))
        return self._deduplicate(evidence)

    def _from_provided_documents(self, documents: Sequence[DocumentPayload]) -> list[SourceEvidence]:
        output: list[SourceEvidence] = []
        for document in documents:
            if document.url is None:
                continue
            source = document.source or self._extract_domain(str(document.url))
            base_confidence = (
                document.confidence_hint
                if document.confidence_hint is not None
                else (0.85 if document.verified_origin else 0.6)
            )
            output.append(
                SourceEvidence(
                    source=source,
                    url=document.url,
                    title=document.title or "Supporting document",
                    excerpt=document.excerpt,
                    content=document.content or document.excerpt,
                    published_at=document.published_at,
                    verified_origin=document.verified_origin,
                    confidence=min(1.0, max(0.0, base_confidence)),
                    raw_score=min(1.0, max(0.0, base_confidence)),
                )
            )
        return output

    @staticmethod
    def _deduplicate(evidence: list[SourceEvidence]) -> list[SourceEvidence]:
        seen_urls: set[str] = set()
        unique: list[SourceEvidence] = []
        for item in evidence:
            url_str = str(item.url)
            if url_str in seen_urls:
                continue
            seen_urls.add(url_str)
            unique.append(item)
        return unique

    async def _query_fact_check_api(self, query: str) -> list[SourceEvidence]:
        if not self._settings.fact_check_api_key:
            return []
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"query": query, "key": self._settings.fact_check_api_key}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
        claims = payload.get("claims", [])
        output: list[SourceEvidence] = []
        for item in claims:
            review = item.get("claimReview", [{}])[0]
            link = review.get("url")
            if not link:
                continue
            output.append(
                SourceEvidence(
                    source=self._extract_domain(link),
                    url=link,
                    title=review.get("publisher", {}).get("name", "Fact-check"),
                    excerpt=review.get("text"),
                    content=review.get("text"),
                    published_at=self._parse_datetime(review.get("reviewDate")),
                    verified_origin=True,
                    confidence=0.9,
                    raw_score=0.9,
                )
            )
        return output

    async def _query_news_api(self, query: str) -> list[SourceEvidence]:
        if not self._settings.news_api_key:
            return []
        url = "https://newsapi.org/v2/everything"
        params = {"q": query, "pageSize": 5, "language": "en"}
        headers = {"Authorization": self._settings.news_api_key}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()
        articles = payload.get("articles", [])
        output: list[SourceEvidence] = []
        for article in articles:
            link = article.get("url")
            if not link:
                continue
            output.append(
                SourceEvidence(
                    source=self._extract_domain(link),
                    url=link,
                    title=article.get("title", "Verified article"),
                    excerpt=article.get("description"),
                    content=article.get("content") or article.get("description"),
                    published_at=self._parse_datetime(article.get("publishedAt")),
                    verified_origin=False,
                    confidence=0.7,
                    raw_score=0.7,
                )
            )
        return output

    async def _query_search_api(self, query: str, url: str | None) -> list[SourceEvidence]:
        if not self._settings.search_api_key or not self._settings.search_engine_id:
            return []
        api_url = "https://customsearch.googleapis.com/customsearch/v1"
        params: dict[str, str] = {
            "q": query,
            "key": self._settings.search_api_key,
            "cx": self._settings.search_engine_id,
        }
        if url:
            params["exactTerms"] = url
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(api_url, params=params)
            response.raise_for_status()
            payload = response.json()
        items = payload.get("items", [])
        output: list[SourceEvidence] = []
        for item in items[:5]:
            link = item.get("link")
            if not link:
                continue
            output.append(
                SourceEvidence(
                    source=self._extract_domain(link),
                    url=link,
                    title=item.get("title", "Search result"),
                    excerpt=item.get("snippet"),
                    content=item.get("snippet"),
                    published_at=None,
                    verified_origin=False,
                    confidence=0.4,
                    raw_score=0.4,
                )
            )
        return output

    async def _query_duckduckgo_news(self, query: str) -> list[SourceEvidence]:
        return await asyncio.to_thread(self._ddg_news_sync, query)

    def _ddg_news_sync(self, query: str) -> list[SourceEvidence]:
        output: list[SourceEvidence] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.news(query, max_results=6, safesearch="moderate", region="wt-wt"):
                    link = item.get("url")
                    if not link:
                        continue
                    source = item.get("source") or self._extract_domain(link)
                    content_text, summary = self._extract_article_content(link)
                    excerpt = summary or item.get("body") or item.get("snippet")
                    output.append(
                        SourceEvidence(
                            source=source,
                            url=link,
                            title=item.get("title") or "DuckDuckGo news",
                            excerpt=excerpt,
                            content=content_text or excerpt,
                            published_at=None,
                            verified_origin=False,
                            confidence=0.55,
                            raw_score=0.55,
                        )
                    )
        except Exception:
            return []
        return output

    async def _query_google_rss(self, query: str) -> list[SourceEvidence]:
        return await asyncio.to_thread(self._google_rss_sync, query)

    def _google_rss_sync(self, query: str) -> list[SourceEvidence]:
        encoded_query = quote_plus(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        entries = getattr(feed, "entries", [])
        output: list[SourceEvidence] = []
        for entry in entries[:6]:
            link = entry.get("link")
            if not link:
                continue
            title = entry.get("title")
            published = entry.get("published") or entry.get("updated")
            content_text, summary = self._extract_article_content(link)
            output.append(
                SourceEvidence(
                    source=self._extract_domain(link),
                    url=link,
                    title=title or "Google News result",
                    excerpt=summary or entry.get("summary"),
                    content=content_text or summary or entry.get("summary"),
                    published_at=self._parse_rfc_datetime(published),
                    verified_origin=False,
                    confidence=0.6,
                    raw_score=0.6,
                )
            )
        return output

    def _extract_domain(self, url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc or url

    def _parse_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _parse_rfc_datetime(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None

    def _extract_article_content(self, url: str) -> tuple[str | None, str | None]:
        try:
            downloaded = trafilatura.fetch_url(url, timeout=self._timeout)
        except Exception:
            return None, None
        if not downloaded:
            return None, None
        json_payload: dict[str, object] | None = None
        try:
            extracted_json = trafilatura.extract(downloaded, output_format="json")
            if extracted_json:
                json_payload = json.loads(extracted_json)
        except (ValueError, json.JSONDecodeError):
            json_payload = None
        if isinstance(json_payload, dict):
            text = json_payload.get("text") or json_payload.get("raw_text")
            summary = json_payload.get("summary") or json_payload.get("title")
            text_str = str(text) if text else None
            summary_str = str(summary) if summary else None
            if text_str:
                if summary_str is None and text_str:
                    summary_str = text_str[:400]
                return text_str, summary_str
        try:
            text_fallback = trafilatura.extract(downloaded)
        except Exception:
            text_fallback = None
        if text_fallback:
            text_fallback = str(text_fallback)
            summary_fallback = text_fallback[:400]
            return text_fallback, summary_fallback
        return None, None
