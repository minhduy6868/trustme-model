# TrustScore Verification Service

This prototype demonstrates how to orchestrate on-demand verification of online claims by querying reputable, verified news sources and reputation feeds. When a user submits a claim or URL, the service searches curated sources, gauges publisher credibility, and calculates a trust score.

## Key Ideas

- **On-demand scans**: No persistent preload. Each request triggers fresh lookups against external APIs.
- **Evidence collection**: Queries are proxied through reputable search and fact-check APIs (Google Fact Check Tools, Custom Search, News APIs) instead of scraping the open web directly.
- **Trust scoring**: Evidence is weighted by publisher reliability, semantic agreement, language style risk, and media verification. Scores under 0.85 trigger alternative credible links.
- **Extensible signals**: Text, metadata, optional media hashes, and reverse-image lookups all feed the scoring pipeline.

## Project Layout

```
.
├── README.md
├── requirements.txt
└── src
    ├── main.py
    └── trustscore
        ├── __init__.py
        ├── config.py
        ├── models.py
        ├── reputation.py
        ├── sources.py
        └── trust_engine.py
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Export API credentials for the upstream services you plan to use. The sample config expects these environment variables:

   - `FACT_CHECK_API_KEY`
   - `NEWS_API_KEY`
   - `SEARCH_API_KEY`
   - `SEARCH_ENGINE_ID`
   - `IMAGE_SEARCH_API_KEY`
   - `IMAGE_SEARCH_API_URL`
   - `NEWSDATA_API_KEY`
   - `OPENPAGERANK_API_KEY`

4. Run the FastAPI service:

   ```bash
   uvicorn src.main:app --reload
   ```

   Visit `http://127.0.0.1:8000/docs` for an interactive Swagger UI.

## Request Flow

1. **Submit a claim** via `POST /verify` with the claim text, optional URL/context, and optional media hashes.
2. **Source scanning** concurrently queries verified APIs:
   - fact-check feeds (e.g., Google Fact Check Tools)
   - curated news APIs (e.g., NewsAPI)
   - Custom Search results limited to trusted publishers
3. **Signal extraction** enriches the claim with:
   - publisher reputation (via a reputation service or local domain lists)
   - semantic agreement using Sentence Transformers
   - language-style risk heuristics (detect clickbait/emotive language)
   - reverse-image confidence if hashes are supplied
4. **Evidence synthesis** combines the weighted signals into a trust score between 0 and 1. When the score is below 0.85, the response includes alternative reputable articles for further reading.

## Limitations & Next Steps

- External APIs enforce rate limits and access constraints. Implement caching or queueing for burst traffic.
- Image and multimedia verification require dedicated perceptual hashing pipelines (e.g., Google Lens API, AWS Rekognition). The included `ImageVerifier` assumes an upstream reverse-image API and returns neutral scores if none is configured.
- True end-to-end "internet-wide" coverage is infeasible; focus on trusted, auditable sources and disclose confidence and evidence provenance transparently.
- Add persistence only for audit logs per user request if compliance requires.

## Security & Compliance

- Store API keys securely (e.g., secret manager or environment variables).
- Log evidence provenance to defend trust scores.
- Respect robots.txt and terms of service for each upstream data provider.
