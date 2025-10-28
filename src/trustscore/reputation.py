from __future__ import annotations

import asyncio

import httpx
import tldextract

from .config import get_settings
from .models import ReputationProfile


class ReputationClient:
    def __init__(self, *, timeout: float = 8.0) -> None:
        self._settings = get_settings()
        self._timeout = timeout
        self._cache: dict[str, ReputationProfile] = {}

    async def fetch(self, source: str) -> ReputationProfile:
        normalized = self._normalize_domain(source)
        if normalized in self._cache:
            return self._cache[normalized]
        profile = await self._fetch_with_sources(normalized)
        self._cache[normalized] = profile
        return profile

    async def batch_fetch(self, sources: list[str]) -> dict[str, ReputationProfile]:
        tasks = [self.fetch(source) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            self._normalize_domain(source): result
            if isinstance(result, ReputationProfile)
            else self._default_profile(source)
            for source, result in zip(sources, results, strict=False)
        }

    async def _fetch_with_sources(self, domain: str) -> ReputationProfile:
        newsdata_profile = await self._query_newsdata_api(domain)
        if newsdata_profile:
            return newsdata_profile
        opr_profile = await self._query_openpagerank(domain)
        if opr_profile:
            return opr_profile
        return self._default_profile(domain)

    async def _query_newsdata_api(self, domain: str) -> ReputationProfile | None:
        api_key = self._settings.newsdata_api_key
        if not api_key:
            return None
        endpoint = "https://newsdata.io/api/1/source"
        params = {"apikey": api_key, "domain": domain}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError:
                return None
        if not isinstance(payload, dict):
            return None
        sources = payload.get("results", [])
        if not sources:
            return None
        best = sources[0]
        reliability = float(best.get("reliability", 0.5))
        top_source = bool(best.get("top_source", False))
        return ReputationProfile(
            source=domain,
            score=max(0.0, min(1.0, reliability)),
            verified=top_source,
            metadata={"provider": "newsdata", **best},
        )

    async def _query_openpagerank(self, domain: str) -> ReputationProfile | None:
        api_key = self._settings.openpagerank_api_key
        if not api_key:
            return None
        endpoint = "https://openpagerank.com/api/v1.0/getPageRank"
        params = {"domains[0]": domain}
        headers = {"API-OPR": api_key}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(endpoint, params=params, headers=headers)
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError:
                return None
        if not isinstance(payload, dict):
            return None
        results = payload.get("response")
        if not isinstance(results, list) or not results:
            return None
        entry = results[0]
        rank = float(entry.get("page_rank_decimal", 0.4))
        rank_position = entry.get("rank", 0)
        return ReputationProfile(
            source=domain,
            score=max(0.0, min(1.0, rank)),
            verified=bool(rank_position and rank_position <= 10000),
            metadata={"provider": "openpagerank", **entry},
        )

    def _normalize_domain(self, source: str) -> str:
        extracted = tldextract.extract(source)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
        return source.lower()

    def _default_profile(self, source: str) -> ReputationProfile:
        return ReputationProfile(
            source=self._normalize_domain(source),
            score=0.5,
            verified=False,
            metadata={"provider": "default"},
        )
