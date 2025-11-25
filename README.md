# TrustMe Model API

AI-powered fact-checking and verification engine.

## Purpose

Main API that orchestrates fact-checking:
- Receives requests from Chrome Extension
- Calls Crawler API to gather data
- Runs AI verification (7 detection methods)
- Returns trust score and verdict

## Features

### Core Verification

- Spam language detection (clickbait, sensational phrases)
- Content duplication checking
- Authority verification (trusted sources)
- Spam behavior detection (same account/domain)
- Fact extraction & verification (numbers, dates, names)
- External fact-checking APIs (Google Fact Check Tools)
- Donation scam detection (fake charity posts)

### Production Features

- Rate limiting (60 req/min per IP)
- Content caching (1 hour)
- Concurrent job limiting (50 max)
- Redis integration with memory fallback
- Full logging (console + file)
- Metrics tracking
- Health checks
- Graceful shutdown

## Installation

### Basic:
```bash
pip install -r requirements.txt
```

### Advanced (with NLI model):
```bash
pip install -r requirements_improved.txt
```

### Redis (recommended):
```bash
# macOS
brew install redis
redis-server

# Docker
docker run -d -p 6379:6379 redis:alpine
```

## Configuration

Optional `.env` file:

```bash
CRAWLER_API_URL=http://localhost:8000
REDIS_URL=redis://localhost:6379
MAX_CONCURRENT_JOBS=50
DATA_DIR=./trustme-model/data

# Optional: Google Fact Check API
FACT_CHECK_API_KEY=your_key_here
```

## Run

```bash
uvicorn src.main:app --port 8001 --reload
```

API runs on **http://localhost:8001**

## API Endpoints

### POST /verify

Submit verification request (async).

**Request:**
```json
{
  "text": "Content to verify",
  "url": "https://example.com",
  "language": "vi",
  "deep_analysis": true
}
```

**Response (immediate):**
```json
{
  "job_id": "job_abc123",
  "status": "processing"
}
```

### GET /result/{job_id}

Get verification result (poll every 2s).

**Response (when completed):**
```json
{
  "job_id": "job_abc123",
  "status": "completed",
  "trust_score": 85.5,
  "verdict": "verified",
  "summary": "Found on 3 trusted sources",
  "is_donation_post": false,
  "components": {
    "spam_detection": 92.0,
    "authority": 85.0,
    "duplication": 75.0,
    "fact_check": 88.0
  },
  "alternatives": [...]
}
```

### GET /health

Health check with component status.

### GET /metrics

Service metrics (requests, success rate, uptime).

## Project Structure

```
trustme-model/
├── config/
│   ├── spam_patterns.json    # Detection patterns (customizable)
│   └── settings.py           # Configuration
├── logs/
│   └── trustscore.log        # Service logs
├── scripts/
│   ├── test_api.py
│   └── check_dependencies.py
├── src/
│   ├── main.py               # Main API
│   └── trustscore/           # Core logic
│       ├── realtime_verifier.py
│       ├── donation_detector.py
│       ├── fact_extractor.py
│       ├── external_verifier.py
│       └── ...
└── tests/
    ├── test_api.py
    └── test_storage.py
```

## Customization

Detection datasets live in `trustme-model/data/` (regex patterns, official accounts, phishing domains, fact-check DB, etc.).

- Edit JSON files in `data/` then recompile with `python scripts/compile_datasets.py --data-dir trustme-model/data`.
- `DATA_DIR` env var controls which folder is loaded at startup.

## Accuracy

- Obvious fake news: 80-90%
- Real news from trusted sources: 85-95%
- Donation scams: 75-85%
- Subtle misinformation: 50-60%
- **Average: 70-75%**
- **With Google Fact Check API: 75-85%**

## Performance

- Processing time: 10-30s per request
- Max concurrent jobs: 50 (configurable)
- Rate limit: 60 requests/min per IP
- Cache hit rate: ~30-50% (with Redis)

## Testing

```bash
# Test logic only
python test_logic_only.py

# Test API
python scripts/test_api.py

# Check dependencies
python scripts/check_dependencies.py
```

## Dependencies

- Crawler API (port 8000) must be running
- Redis (optional but recommended for production)

## Documentation

- API docs: http://localhost:8001/docs
- Setup guide: See `/SETUP.md` in root
- Full documentation: See `/HUONG_DAN.md` in root

## License

Apache 2.0
