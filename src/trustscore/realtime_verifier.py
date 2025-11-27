"""
Realtime Verification Engine
Rule-based + heuristics approach powered by external datasets.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from data_loader import load_datasets
from trustscore.vector_fusion import MultiVectorFusion, FusionWeights, EmbeddingBackbone


def _safe_parse_datetime(value: Any) -> Optional[datetime]:
    """Parse ISO-like datetimes safely."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _clamp_score(score: float) -> float:
    return max(0.0, min(100.0, score))


def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"https?://[^\s)]+", text)


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def _split_claims(text: str, max_sentences: int = 4) -> List[str]:
    if not text:
        return []
    parts = re.split(r"[\.!\?]\s+", text)
    sentences = [p.strip() for p in parts if len(p.strip()) > 15]
    return sentences[:max_sentences]


def _primary_domain(found_articles: List[Dict[str, Any]], meta: Optional[Dict[str, Any]]) -> str:
    if meta and isinstance(meta.get("domain_frequency"), dict):
        if meta["domain_frequency"]:
            return max(meta["domain_frequency"].items(), key=lambda kv: kv[1])[0] or ""
    for article in found_articles:
        if article.get("domain"):
            return str(article["domain"])
    return ""


def _is_whitelisted(domain: str, whitelist: List[str]) -> bool:
    if not domain:
        return False
    domain_lower = domain.lower()
    return any(domain_lower.endswith(w.lower()) or w.lower() in domain_lower for w in whitelist)


def _has_donation_signals(text: str, donation_indicators: List[str], bank_patterns: List[re.Pattern]) -> bool:
    if not text:
        return False
    lower_text = text.lower()
    if any(indicator.lower() in lower_text for indicator in donation_indicators):
        return True
    for pattern in bank_patterns:
        if pattern.search(text):
            return True
    return False


def _reference_texts(found_articles: List[Dict[str, Any]], max_refs: int = 12) -> List[str]:
    refs: List[str] = []
    for item in found_articles:
        title = item.get("title") or ""
        snippet = item.get("snippet") or ""
        content = item.get("article") or item.get("content") or ""
        combined = " ".join([title, snippet, content]).strip()
        if combined:
            refs.append(combined)
        if len(refs) >= max_refs:
            break
    return refs


def _looks_invalid_account(account: str) -> bool:
    """Return True for obviously fake banking accounts (too short/long or repeating digits)."""
    if not account:
        return True
    if len(account) < 9 or len(account) > 16:
        return True
    return len(set(account)) == 1


@dataclass
class DetectorResult:
    name: str
    score_delta: float
    confidence: float
    evidence: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    override: bool = False
    verdict_override: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def detect_clickbait(text: str, patterns: List[re.Pattern], language: str = "vi") -> DetectorResult:
    matches: List[str] = []
    for pattern in patterns:
        matches.extend([m.group(0) for m in pattern.finditer(text)])

    speculative_markers = [
        "tin đồn",
        "đồn đoán",
        "có thể",
        "dường như",
        "reportedly",
        "rumor",
        "alleged",
        "không ai ngờ",
    ]
    spec_hits = [w for w in speculative_markers if w in text.lower()]
    official_claim = "thông báo chính thức" in text.lower()

    if not matches and not spec_hits and not official_claim:
        return DetectorResult(name="clickbait", score_delta=0, confidence=0.25, flags=["content"])

    penalty = min(25.0, len(matches) * 5.0 + len(spec_hits) * 3.0)
    if official_claim and (matches or text.count("!") >= 2):
        penalty += 5
        spec_hits.append("official-tone-mismatch")

    evidence = [f"pattern: {m}" for m in matches[:4]]
    evidence.extend([f"speculative: {w}" for w in spec_hits[:3]])

    return DetectorResult(
        name="clickbait",
        score_delta=-penalty,
        confidence=0.85,
        evidence=evidence,
        flags=["clickbait", "speculative"] if spec_hits else ["clickbait"],
    )


def detect_authority(
    found_articles: List[Dict[str, Any]],
    domain_whitelist: List[str],
    primary_domain: Optional[str] = None,
    trusted_pages: Optional[List[Dict[str, Any]]] = None,
    trusted_sources: Optional[List[Dict[str, Any]]] = None,
) -> DetectorResult:
    trusted = []
    gov_hits = []
    primary_hit = False
    page_hits: List[str] = []
    trusted_pages = trusted_pages or []
    trusted_sources = trusted_sources or []

    primary_entry = None
    if primary_domain:
        primary_entry = next(
            (s for s in trusted_sources if (s.get("identifier") or "").lower() == primary_domain.lower()), None
        )

    if primary_domain:
        domain = primary_domain.lower()
        if any(domain.endswith(w.lower()) or w.lower() in domain for w in domain_whitelist):
            trusted.append(domain)
            primary_hit = True
            if domain.endswith(".gov") or domain.endswith(".gov.vn") or "gov" in domain:
                gov_hits.append(domain)

    for article in found_articles:
        domain = (article.get("domain") or "").lower()
        if any(domain.endswith(w.lower()) or w.lower() in domain for w in domain_whitelist):
            trusted.append(domain)
            if domain.endswith(".gov") or domain.endswith(".gov.vn") or "gov" in domain:
                gov_hits.append(domain)

        url = (article.get("url") or "").lower()
        for page in trusted_pages:
            ident = (page.get("identifier") or "").lower()
            if ident and ident in url:
                page_hits.append(page.get("name") or ident)
                trusted.append(ident)

    if not trusted:
        return DetectorResult(name="authority", score_delta=0, confidence=0.2, flags=["authority"])

    base_boost = 80.0 if primary_hit else 0.0
    boost = base_boost + min(30.0, 8.0 * len(set(trusted)))
    if page_hits:
        page_bonus = min(40.0, 12.0 * len(set(page_hits)))
        boost += page_bonus
    if gov_hits:
        boost += 10.0

    evidence = [f"trusted domain: {d}" for d in sorted(set(trusted)) if d not in page_hits][:5]
    if page_hits:
        evidence.extend([f"trusted page: {p}" for p in sorted(set(page_hits))[:3]])
    primary_fullscore = bool(primary_entry and (primary_entry.get("full_score") or primary_entry.get("trust_score", 0) >= 0.95))

    return DetectorResult(
        name="authority",
        score_delta=boost,
        confidence=0.95 if primary_hit else 0.9,
        evidence=evidence,
        flags=["authority", "official"] if gov_hits else ["authority"],
        meta={
            "trusted_domains": sorted(set(trusted)),
            "primary_whitelisted": primary_hit,
            "primary_fullscore": primary_fullscore,
            "trusted_pages": sorted(set(page_hits)),
        },
    )


def cross_source_authority_boost(
    similarity: float,
    found_articles: List[Dict[str, Any]],
    domain_whitelist: List[str],
    primary_domain: Optional[str] = None,
    trusted_pages: Optional[List[Dict[str, Any]]] = None,
) -> DetectorResult:
    whitelist_hits = []
    page_hits = []
    trusted_pages = trusted_pages or []
    for article in found_articles:
        domain = (article.get("domain") or "").lower()
        if not domain or (primary_domain and domain == primary_domain.lower()):
            continue
        if any(domain.endswith(w.lower()) or w.lower() in domain for w in domain_whitelist):
            whitelist_hits.append(domain)
        url = (article.get("url") or "").lower()
        for page in trusted_pages:
            ident = (page.get("identifier") or "").lower()
            if ident and ident in url:
                page_hits.append(page.get("name") or ident)

    unique_hits = sorted(set(whitelist_hits))
    unique_pages = sorted(set(page_hits))
    if similarity >= 0.9 and len(unique_hits) >= 2:
        boost = 80.0
        flags = ["authority-cross", "likely-real"]
    elif similarity >= 0.8 and (unique_hits or unique_pages):
        boost = 55.0
        flags = ["authority-cross"]
    else:
        return DetectorResult(
            name="cross-authority",
            score_delta=0.0,
            confidence=0.4,
            evidence=[],
            flags=["authority"],
        )

    evidence = [f"confirmed by: {d}" for d in unique_hits[:3]]
    evidence.extend([f"confirmed page: {p}" for p in unique_pages[:2]])
    return DetectorResult(
        name="cross-authority",
        score_delta=boost,
        confidence=0.9,
        evidence=evidence,
        flags=flags,
        meta={
            "matched_trusted_domains": unique_hits,
            "matched_trusted_pages": unique_pages,
            "similarity": similarity,
        },
    )


def _fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return hashlib.md5(normalized.encode()).hexdigest()


def _jaccard_similarity(text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def detect_duplication(original: str, found_articles: List[Dict[str, Any]]) -> DetectorResult:
    if not original:
        return DetectorResult(name="duplication", score_delta=0, confidence=0.1, flags=["uniqueness"])

    original_fp = _fingerprint(original)
    similarities: List[float] = []
    exact_hits = 0

    for article in found_articles:
        content = article.get("content") or article.get("snippet") or ""
        if not content:
            continue
        fp = _fingerprint(content)
        if fp == original_fp:
            exact_hits += 1
            similarities.append(1.0)
            continue
        similarities.append(_jaccard_similarity(original, content))

    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    high_sim = [s for s in similarities if s > 0.95]

    if not similarities:
        return DetectorResult(name="duplication", score_delta=0, confidence=0.35, flags=["uniqueness"])

    if exact_hits or high_sim:
        evidence = [f"exact duplicates: {exact_hits}", f"high similarity: {len(high_sim)}", f"avg similarity: {avg_sim:.2f}"]
        penalty = min(25.0, 10.0 * (exact_hits + len(high_sim)))
        return DetectorResult(
            name="duplication",
            score_delta=-penalty,
            confidence=0.88,
            evidence=evidence,
            flags=["duplication"],
            meta={"average_similarity": avg_sim},
        )

    return DetectorResult(
        name="duplication",
        score_delta=5.0 if avg_sim > 0.4 else 0.0,
        confidence=0.5,
        evidence=[f"average similarity: {avg_sim:.2f}"],
        flags=["uniqueness"],
        meta={"average_similarity": avg_sim},
    )


def detect_single_source(found_articles: List[Dict[str, Any]]) -> DetectorResult:
    domains = {article.get("domain") for article in found_articles if article.get("domain")}
    if len(domains) <= 1:
        return DetectorResult(
            name="single-source",
            score_delta=-20.0,
            confidence=0.8,
            evidence=["Only one unique domain found"],
            flags=["single-source"],
        )
    return DetectorResult(name="single-source", score_delta=0, confidence=0.4, flags=["single-source"])


def detect_spam_behavior(found_articles: List[Dict[str, Any]]) -> DetectorResult:
    domain_counts = Counter()
    author_counts = Counter()
    for article in found_articles:
        domain_counts[(article.get("domain") or "unknown").lower()] += 1
        author = (article.get("author") or "").strip().lower()
        if author:
            author_counts[author] += 1

    max_domain = max(domain_counts.values()) if domain_counts else 0
    max_author = max(author_counts.values()) if author_counts else 0
    if max_domain <= 3 and max_author <= 2:
        return DetectorResult(name="spam-behavior", score_delta=0, confidence=0.35, flags=["behaviour"])

    evidence = [f"top domain posts: {max_domain}", f"top author posts: {max_author}"]
    penalty = min(30.0, (max_domain - 3) * 5.0 + (max_author - 2) * 4.0)
    return DetectorResult(
        name="spam-behavior",
        score_delta=-penalty,
        confidence=0.8,
        evidence=evidence,
        flags=["spam-behavior"],
        meta={"domain_counts": domain_counts},
    )


def detect_donation(
    text: str,
    official_accounts: Dict[str, Any],
    bank_account_patterns: List[re.Pattern],
    donation_indicators: List[str],
    donation_scam_patterns: Optional[List[re.Pattern]] = None,
) -> DetectorResult:
    lower_text = text.lower()
    is_donation = any(indicator.lower() in lower_text for indicator in donation_indicators)

    accounts: List[str] = []
    for pattern in bank_account_patterns:
        for match in pattern.finditer(text):
            # Use last group if exists, otherwise full match
            value = match.group(match.lastindex or 0)
            accounts.append(value)

    invalid_accounts = [acc for acc in accounts if _looks_invalid_account(acc)]

    scam_pattern_hits: List[str] = []
    for pattern in donation_scam_patterns or []:
        for m in pattern.finditer(text):
            scam_pattern_hits.append(m.group(0))

    mentioned_orgs = []
    for org_key, org_accounts in official_accounts.items():
        org_name = org_key.lower()
        if org_name in lower_text or any((acc.get("name") or "").lower() in lower_text for acc in org_accounts):
            mentioned_orgs.append(org_key)

    if not is_donation and not accounts:
        return DetectorResult(name="donation", score_delta=0, confidence=0.2, flags=["donation"])

    evidence = []
    flags = ["donation"]
    score_delta = -10.0 if is_donation else 0.0
    override = False
    verdict_override = None

    if accounts:
        evidence.append(f"accounts detected: {', '.join(accounts[:3])}")

    # Check against official accounts
    verified_accounts = []
    fake_accounts = []
    for org in mentioned_orgs:
        official_list = official_accounts.get(org, [])
        allows_unknown = any(item.get("account") == "*" for item in official_list)
        official_numbers = {item.get("account") for item in official_list if item.get("account") not in (None, "*")}
        if allows_unknown and not official_numbers:
            # Org acknowledged but not enforcing account match to avoid false positives
            continue
        if not official_numbers:
            continue
        for acc in accounts:
            if acc in official_numbers:
                verified_accounts.append((org, acc))
            elif allows_unknown:
                continue
            else:
                fake_accounts.append((org, acc))

    if invalid_accounts:
        override = True
        verdict_override = "donation-scam"
        flags.append("donation-scam-critical")
        evidence.append(f"invalid account format: {invalid_accounts[0]}")
        score_delta -= 35.0
    elif fake_accounts:
        evidence.append(f"mismatched account for {fake_accounts[0][0]}: {fake_accounts[0][1]}")
        if scam_pattern_hits:
            override = True
            verdict_override = "donation-scam"
            flags.append("donation-scam-critical")
            evidence.append(f"scam language: {scam_pattern_hits[0]}")
            score_delta -= 40.0
        else:
            score_delta -= 20.0
            flags.append("donation-unverified-account")
    elif verified_accounts:
        score_delta += 15.0
        flags.append("official-donation")
        evidence.append(f"verified account {verified_accounts[0][1]} for {verified_accounts[0][0]}")
    elif accounts and not mentioned_orgs:
        # Personal appeal with account present
        score_delta -= 20.0
        flags.append("personal-donation")
        evidence.append("personal account without verified organization")

    if is_donation and not accounts:
        evidence.append("donation language without banking details")
        score_delta -= 5.0

    return DetectorResult(
        name="donation",
        score_delta=score_delta,
        confidence=0.75 if accounts else 0.5,
        evidence=evidence,
        flags=flags,
        override=override,
        verdict_override=verdict_override,
        meta={"is_donation_post": is_donation, "accounts": accounts, "mentioned_orgs": mentioned_orgs},
    )


def detect_temporal_manipulation(text: str, found_articles: List[Dict[str, Any]]) -> DetectorResult:
    recency_markers = ["hôm nay", "vừa xảy ra", "breaking today", "ngay lúc này", "just happened"]
    has_recency_claim = any(marker in text.lower() for marker in recency_markers)

    timestamps: List[datetime] = []
    for article in found_articles:
        dt = _safe_parse_datetime(article.get("first_seen") or article.get("published_time") or article.get("created_at"))
        if dt:
            timestamps.append(dt)

    if not timestamps:
        return DetectorResult(name="temporal", score_delta=0, confidence=0.25, flags=["temporal"])

    earliest = min(timestamps)
    delta_days = (datetime.utcnow() - earliest).days
    if has_recency_claim and delta_days > 60:
        evidence = [f"recency wording with old content ({delta_days} days old)"]
        return DetectorResult(
            name="temporal",
            score_delta=-25.0,
            confidence=0.82,
            evidence=evidence,
            flags=["old-news-reposted"],
            override=True,
            verdict_override="likely-false",
        )

    if delta_days > 365:
        return DetectorResult(
            name="temporal",
            score_delta=-10.0,
            confidence=0.6,
            evidence=[f"old content detected ({delta_days} days)"],
            flags=["old-news"],
        )

    return DetectorResult(name="temporal", score_delta=0, confidence=0.3, flags=["temporal"])


def detect_domain_similarity(found_articles: List[Dict[str, Any]], domain_whitelist: List[str], domain_aliases: Dict[str, List[str]]) -> DetectorResult:
    suspicious: List[str] = []
    whitelist_lower = [d.lower() for d in domain_whitelist]

    for article in found_articles:
        domain = (article.get("domain") or "").lower()
        if not domain or domain in whitelist_lower:
            continue

        close = difflib.get_close_matches(domain, whitelist_lower, n=1, cutoff=0.82)
        alias_hit = any(domain == alias.lower() for aliases in domain_aliases.values() for alias in aliases)
        if close or alias_hit:
            suspicious.append(domain)

    if not suspicious:
        return DetectorResult(name="domain-similarity", score_delta=0, confidence=0.2, flags=["domain"])

    evidence = [f"suspicious domain: {d}" for d in sorted(set(suspicious))[:5]]
    return DetectorResult(
        name="domain-similarity",
        score_delta=-15.0,
        confidence=0.78,
        evidence=evidence,
        flags=["typosquatting"],
    )


def detect_external_fact_check(text: str, fact_check_db: List[Dict[str, Any]]) -> DetectorResult:
    if not fact_check_db:
        return DetectorResult(name="external-fact-check", score_delta=0, confidence=0.1, flags=["fact-check"])

    best_match = None
    best_ratio = 0.0
    for entry in fact_check_db:
        claim_text = entry.get("claim_text", "")
        ratio = difflib.SequenceMatcher(None, text.lower(), claim_text.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = entry

    if not best_match or best_ratio < 0.78:
        return DetectorResult(name="external-fact-check", score_delta=0, confidence=0.35, flags=["fact-check"])

    verdict = best_match.get("verdict")
    evidence = [f"matched claim '{best_match.get('claim_hash')}' ({best_ratio:.2f}) -> {verdict}"]

    if verdict == "false":
        return DetectorResult(
            name="external-fact-check",
            score_delta=-35.0,
            confidence=0.92,
            evidence=evidence,
            flags=["debunked"],
            override=True,
            verdict_override="likely-false",
        )
    if verdict == "true":
        return DetectorResult(
            name="external-fact-check",
            score_delta=20.0,
            confidence=0.92,
            evidence=evidence,
            flags=["fact-verified"],
        )
    if verdict == "mixture":
        return DetectorResult(
            name="external-fact-check",
            score_delta=-10.0,
            confidence=0.7,
            evidence=evidence,
            flags=["mixed-claim"],
        )

    return DetectorResult(name="external-fact-check", score_delta=0, confidence=0.4, evidence=evidence, flags=["fact-check"])


def detect_ai_generated(text: str, ai_patterns: List[re.Pattern]) -> DetectorResult:
    matches = []
    for pattern in ai_patterns:
        matches.extend([m.group(0) for m in pattern.finditer(text)])

    words = text.split()
    if not words:
        return DetectorResult(name="ai-content", score_delta=0, confidence=0.1, flags=["ai"])

    word_counts = Counter(words)
    repetition_ratio = max(word_counts.values()) / len(words)
    sentence_lengths = [len(s.split()) for s in re.split(r"[.?!]", text) if s.strip()]
    uniformity = 0.0
    if sentence_lengths:
        avg_len = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((l - avg_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        uniformity = 1.0 / (1.0 + variance)

    hedging_markers = ["có thể", "nghe nói", "theo nguồn tin", "reportedly", "rumor", "allegedly"]
    sentences = [s.strip() for s in re.split(r"[.?!]", text) if s.strip()]
    hedging_hits = sum(1 for s in sentences if any(marker in s.lower() for marker in hedging_markers))

    suspicion_score = 0.0
    if matches:
        suspicion_score += 0.4
    if repetition_ratio > 0.08:
        suspicion_score += 0.3
    if uniformity > 0.25:
        suspicion_score += 0.2
    if hedging_hits:
        suspicion_score += min(0.2, hedging_hits * 0.05)

    if suspicion_score < 0.4:
        return DetectorResult(name="ai-content", score_delta=0, confidence=0.3, flags=["ai"])

    evidence = [f"repetition ratio {repetition_ratio:.2f}", f"uniformity {uniformity:.2f}"]
    if matches:
        evidence.append(f"patterns: {', '.join(set(matches[:2]))}")
    if hedging_hits:
        evidence.append(f"hedging sentences: {hedging_hits}")

    return DetectorResult(
        name="ai-content",
        score_delta=-15.0,
        confidence=0.8,
        evidence=evidence,
        flags=["ai-generated"],
    )


def detect_media_age(image_urls: List[str], found_articles: List[Dict[str, Any]]) -> DetectorResult:
    if not image_urls:
        return DetectorResult(name="media-age", score_delta=0, confidence=0.1, flags=["media"])

    reused: List[str] = []
    for article in found_articles:
        if article.get("screenshot_hash") and any(
            article.get("screenshot_hash") == other.get("screenshot_hash")
            for other in found_articles
            if other is not article
        ):
            reused.append(article.get("screenshot_hash"))

    if not reused:
        return DetectorResult(name="media-age", score_delta=0, confidence=0.3, flags=["media"])

    evidence = [f"reused media hash: {h}" for h in set(reused)]
    return DetectorResult(
        name="media-age",
        score_delta=-12.0,
        confidence=0.7,
        evidence=evidence,
        flags=["old-media"],
    )


def detect_translation_issues(text: str, translation_patterns: List[re.Pattern]) -> DetectorResult:
    hits = []
    for pattern in translation_patterns:
        hits.extend([m.group(0) for m in pattern.finditer(text)])
    if not hits:
        return DetectorResult(name="translation", score_delta=0, confidence=0.15, flags=["translation"])
    return DetectorResult(
        name="translation",
        score_delta=-8.0,
        confidence=0.65,
        evidence=[f"translation marker: {h}" for h in hits[:3]],
        flags=["translation-warning"],
    )


def detect_phishing_or_malware_links(text: str, phishing_domains: List[str], found_articles: List[Dict[str, Any]]) -> DetectorResult:
    urls = _extract_urls(text)
    for article in found_articles:
        if article.get("url"):
            urls.append(article["url"])

    suspicious = []
    for url in urls:
        domain = _domain_from_url(url).lower()
        if domain in (d.lower() for d in phishing_domains):
            suspicious.append(domain)

    if not suspicious:
        return DetectorResult(name="phishing", score_delta=0, confidence=0.3, flags=["safety"])

    evidence = [f"phishing/malware domain: {d}" for d in sorted(set(suspicious))]
    return DetectorResult(
        name="phishing",
        score_delta=-40.0,
        confidence=0.95,
        evidence=evidence,
        flags=["phishing-link"],
        override=True,
        verdict_override="likely-false",
    )


def detect_reports_and_signals(meta: Dict[str, Any]) -> DetectorResult:
    if not meta:
        return DetectorResult(name="reports", score_delta=0, confidence=0.1, flags=["signals"])

    user_reports = meta.get("user_reports", 0)
    share_count = meta.get("share_count", 0)
    social_anomaly = meta.get("social_signals", 0)

    evidence = []
    penalty = 0.0
    if user_reports and user_reports > 5:
        evidence.append(f"{user_reports} user reports")
        penalty += 10.0
    if share_count and share_count > 10000:
        evidence.append(f"high share count: {share_count}")
        penalty += 5.0
    if social_anomaly and social_anomaly < -2:
        evidence.append("negative social signals")
        penalty += 10.0

    if not evidence:
        return DetectorResult(name="reports", score_delta=0, confidence=0.25, flags=["signals"])

    return DetectorResult(
        name="reports",
        score_delta=-penalty,
        confidence=0.7,
        evidence=evidence,
        flags=["reports"],
    )


def detect_fake_event(text: str, found_articles: List[Dict[str, Any]]) -> DetectorResult:
    if len(found_articles) > 2:
        return DetectorResult(name="fake-event", score_delta=0, confidence=0.2, flags=["event"])
    if not text or len(text.split()) < 30:
        return DetectorResult(name="fake-event", score_delta=0, confidence=0.2, flags=["event"])

    evidence = ["Sparse sources and short narrative"]
    return DetectorResult(
        name="fake-event",
        score_delta=-12.0,
        confidence=0.6,
        evidence=evidence,
        flags=["fake-event"],
    )


class RealtimeVerifier:
    """Main realtime verification engine powered by datasets."""

    def __init__(self, api_keys: Optional[Dict[str, str]] = None, datasets: Optional[Dict[str, Any]] = None):
        self.api_keys = api_keys or {}
        self.datasets = datasets or {}
        if not self.datasets:
            self.datasets = load_datasets(
                self._default_data_dir()
            )
        self.compiled = self.datasets.get("_compiled", {})
        # Multi-vector fusion with LaBSE backbone to boost semantic precision
        self.fusion = MultiVectorFusion(embedding_backbone=EmbeddingBackbone())

    def update_datasets(self, datasets: Dict[str, Any]):
        self.datasets = datasets or {}
        self.compiled = self.datasets.get("_compiled", {})

    def _default_data_dir(self) -> str:
        # Fallback to repository default
        return str(Path(__file__).resolve().parents[2] / "data")

    def _get_patterns(self, key: str) -> List[re.Pattern]:
        return list(self.compiled.get(key, []))

    async def verify(
        self,
        original_text: str,
        found_articles: List[Dict[str, Any]],
        language: str = "vi",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        trusted_sources = self.datasets.get("trusted_sources", [])
        domain_whitelist = self.datasets.get("domain_whitelist", [])
        domain_blacklist = self.datasets.get("domain_blacklist", [])
        domain_aliases = self.datasets.get("domain_aliases", {})
        fact_check_db = self.datasets.get("fact_check_db", [])
        phishing_domains = self.datasets.get("phishing_domains", [])
        official_accounts = self.datasets.get("official_accounts", {})
        donation_indicators = self.datasets.get("donation_indicators", [])
        donation_scam_patterns = self._get_patterns("donation_scam_patterns")
        ai_patterns = self._get_patterns("ai_content_patterns")
        translation_patterns = self._get_patterns("translation_warnings")
        bank_patterns = self._get_patterns("bank_account_patterns")

        spam_key = "spam_patterns_vi" if language.lower().startswith("vi") else "spam_patterns_en"
        clickbait_patterns = self._get_patterns(spam_key)

        meta = meta or {}
        detectors: List[DetectorResult] = []

        primary_domain = _primary_domain(found_articles, meta)
        in_whitelist = _is_whitelisted(primary_domain, domain_whitelist)
        in_blacklist = bool(primary_domain) and primary_domain.lower() in {d.lower() for d in domain_blacklist}
        trusted_pages = [s for s in trusted_sources if s.get("type") == "page"]
        trusted_domains = [s for s in trusted_sources if s.get("type") == "domain"]

        # Pre-compute modules needed for authority-first early stop
        temporal_result = detect_temporal_manipulation(original_text, found_articles)
        domain_similarity_result = detect_domain_similarity(found_articles, domain_whitelist, domain_aliases)
        donation_signal_present = _has_donation_signals(original_text, donation_indicators, bank_patterns)

        # 1) Authority check (Authority First)
        authority_result = detect_authority(
            found_articles,
            domain_whitelist,
            primary_domain=primary_domain,
            trusted_pages=trusted_pages,
            trusted_sources=trusted_domains,
        )
        detectors.append(authority_result)
        detectors.append(domain_similarity_result)

        old_news_flag = any(flag in temporal_result.flags for flag in ["old-news", "old-news-reposted"])
        primary_fullscore = authority_result.meta.get("primary_fullscore")
        if (authority_result.meta.get("primary_whitelisted") or primary_fullscore) and not donation_signal_present and "typosquatting" not in domain_similarity_result.flags and not old_news_flag:
            fusion_result = self.fusion.compute(
                original_text,
                found_articles,
                detectors,
                domain_whitelist,
                domain_blacklist,
            )
            trust_score = _clamp_score(100.0)
            evidence = ["FULL SCORE: trusted source (authority-first)"]
            evidence.extend(authority_result.evidence[:3])
            summary = "REAL – Trusted Source (FULL SCORE: trusted source, không phát hiện bất thường)"
            flags = sorted(set(authority_result.flags + domain_similarity_result.flags))
            components = self._components(detectors)
            components["fusion_score"] = fusion_result["fused_score"]
            components.update(
                {
                    "semantic_similarity": fusion_result["semantic_similarity"],
                    "domain_reliability": 1.0,
                    "temporal_consistency": 0.9,
                    "linguistic_certainty": 0.8,
                    "spam_behavior_score": 0.8,
                    "donation_suspicion": 1.0,
                }
            )
            return {
                "trust_score": trust_score,
                "verdict": "verified",
                "explanation": summary,
                "flags": flags,
                "evidence": evidence,
                "confidence_estimate": self._confidence(detectors, trust_score, False),
                "override_reason": "authority-whitelist",
                "details": {
                    "detectors": [asdict(det) for det in detectors],
                    "domain_frequency": Counter([a.get("domain") for a in found_articles if a.get("domain")]),
                    "meta": meta,
                    "fusion": fusion_result,
                },
                "components": components,
                "is_donation_post": False,
            }

        # 2) Semantic fact matching (embeddings)
        claims = _split_claims(original_text)
        references = _reference_texts(found_articles)
        semantic_similarity = self.fusion.embedding.best_similarity(claims, references)
        semantic_evidence = [f"semantic similarity {semantic_similarity:.2f}"]
        if semantic_similarity >= 0.85:
            semantic_delta = 25.0
            semantic_flags = ["semantic-strong"]
        elif semantic_similarity >= 0.5:
            semantic_delta = 5.0
            semantic_flags = ["semantic-medium"]
        else:
            semantic_delta = -25.0 if semantic_similarity < 0.3 else -10.0
            semantic_flags = ["semantic-mismatch"]
        semantic_result = DetectorResult(
            name="semantic-fact-match",
            score_delta=semantic_delta,
            confidence=0.9 if references else 0.5,
            evidence=semantic_evidence,
            flags=semantic_flags,
            meta={"similarity": semantic_similarity, "claims": claims},
        )
        detectors.append(semantic_result)

        # Cross-source authority validation (requires semantic score)
        cross_authority_result = cross_source_authority_boost(
            semantic_similarity,
            found_articles,
            domain_whitelist,
            primary_domain=primary_domain,
            trusted_pages=trusted_pages,
        )
        detectors.append(cross_authority_result)

        # Early stop: semantic mismatch + low-trust domain
        low_trust_domain = (not in_whitelist) and (in_blacklist or len(found_articles) <= 1 or "typosquatting" in domain_similarity_result.flags)
        if semantic_similarity < 0.30 and low_trust_domain:
            fusion_result = self.fusion.compute(
                original_text,
                found_articles,
                detectors,
                domain_whitelist,
                domain_blacklist,
            )
            trust_score = _clamp_score(18.0)
            evidence = ["semantic mismatch <0.30", "low trust domain"]
            evidence.extend(domain_similarity_result.evidence[:2])
            flags = sorted(set(["semantic-mismatch", "low-trust"] + domain_similarity_result.flags))
            components = self._components(detectors)
            components["fusion_score"] = fusion_result["fused_score"]
            components.update(
                {
                    "semantic_similarity": fusion_result["semantic_similarity"],
                    "domain_reliability": 0.2,
                    "temporal_consistency": 0.8,
                    "linguistic_certainty": 0.6,
                    "spam_behavior_score": 0.6,
                    "donation_suspicion": 0.9,
                }
            )
            return {
                "trust_score": trust_score,
                "verdict": "likely-false",
                "explanation": "FAKE / NOT VERIFIED – semantic mismatch mạnh và domain rác",
                "flags": flags,
                "evidence": evidence,
                "confidence_estimate": self._confidence(detectors, trust_score, True),
                "override_reason": "semantic-mismatch-lowtrust",
                "details": {
                    "detectors": [asdict(det) for det in detectors],
                    "domain_frequency": Counter([a.get("domain") for a in found_articles if a.get("domain")]),
                    "meta": meta,
                    "fusion": fusion_result,
                },
                "components": components,
                "is_donation_post": False,
            }

        # 3) External fact check
        fact_check_result = detect_external_fact_check(original_text, fact_check_db)
        detectors.append(fact_check_result)

        # 4) Temporal consistency
        detectors.append(temporal_result)

        # 5) AI-language / stylistic analysis (sentence-level)
        ai_result = detect_ai_generated(original_text, ai_patterns)
        detectors.append(ai_result)

        # 6) Clickbait detection (content quality)
        clickbait_result = detect_clickbait(original_text, clickbait_patterns, language)
        detectors.append(clickbait_result)

        # 7) Spam / duplication signals
        duplication_result = detect_duplication(original_text, found_articles)
        single_source_result = detect_single_source(found_articles)
        spam_result = detect_spam_behavior(found_articles)
        detectors.extend([duplication_result, single_source_result, spam_result])

        # 8) Donation / banking checks (only when signals exist)
        donation_result = detect_donation(
            original_text,
            official_accounts,
            bank_patterns,
            donation_indicators,
            donation_scam_patterns=donation_scam_patterns,
        )
        detectors.append(donation_result)

        # Additional safety/translation/media checks (supporting, not decisive alone)
        image_urls = meta.get("image_urls", []) if meta else []
        translation_result = detect_translation_issues(original_text, translation_patterns)
        media_age_result = detect_media_age(image_urls, found_articles)
        phishing_result = detect_phishing_or_malware_links(original_text, phishing_domains, found_articles)
        reports_result = detect_reports_and_signals(meta or {})
        fake_event_result = detect_fake_event(original_text, found_articles)
        detectors.extend([media_age_result, translation_result, phishing_result, reports_result, fake_event_result])

        # Early stop for old news misleading on low-trust domain
        if temporal_result.override and not in_whitelist:
            fusion_result = self.fusion.compute(
                original_text,
                found_articles,
                detectors,
                domain_whitelist,
                domain_blacklist,
            )
            trust_score = _clamp_score(22.0)
            evidence = temporal_result.evidence[:3] or ["old news resurfaced"]
            evidence.insert(0, "temporal inconsistency")
            components = self._components(detectors)
            components["fusion_score"] = fusion_result["fused_score"]
            components.update(
                {
                    "semantic_similarity": fusion_result["semantic_similarity"],
                    "domain_reliability": 0.35,
                    "temporal_consistency": 0.2,
                    "linguistic_certainty": 0.65,
                    "spam_behavior_score": 0.6,
                    "donation_suspicion": 0.9,
                }
            )
            return {
                "trust_score": trust_score,
                "verdict": temporal_result.verdict_override or "likely-false",
                "explanation": "MISLEADING – Old News",
                "flags": sorted(set(temporal_result.flags + domain_similarity_result.flags)),
                "evidence": evidence,
                "confidence_estimate": self._confidence(detectors, trust_score, True),
                "override_reason": "old-news",
                "details": {
                    "detectors": [asdict(det) for det in detectors],
                    "domain_frequency": Counter([a.get("domain") for a in found_articles if a.get("domain")]),
                    "meta": meta,
                    "fusion": fusion_result,
                },
                "components": components,
                "is_donation_post": donation_result.meta.get("is_donation_post", False),
            }

        # Early stop for donation scam
        if donation_result.override:
            fusion_result = self.fusion.compute(
                original_text,
                found_articles,
                detectors,
                domain_whitelist,
                domain_blacklist,
            )
            trust_score = _clamp_score(15.0)
            evidence = donation_result.evidence[:3] or ["donation scam signals"]
            components = self._components(detectors)
            components["fusion_score"] = fusion_result["fused_score"]
            components.update(
                {
                    "semantic_similarity": fusion_result["semantic_similarity"],
                    "domain_reliability": 0.25,
                    "temporal_consistency": 0.7,
                    "linguistic_certainty": 0.6,
                    "spam_behavior_score": 0.6,
                    "donation_suspicion": 0.0,
                }
            )
            return {
                "trust_score": trust_score,
                "verdict": donation_result.verdict_override or "likely-false",
                "explanation": "SCAM / FRAUD – phát hiện dấu hiệu chuyển khoản bất thường",
                "flags": sorted(set(donation_result.flags)),
                "evidence": evidence,
                "confidence_estimate": self._confidence(detectors, trust_score, True),
                "override_reason": "donation-scam",
                "details": {
                    "detectors": [asdict(det) for det in detectors],
                    "domain_frequency": Counter([a.get("domain") for a in found_articles if a.get("domain")]),
                    "meta": meta,
                    "fusion": fusion_result,
                },
                "components": components,
                "is_donation_post": donation_result.meta.get("is_donation_post", False),
            }

        # Multi-vector fusion (for reference) and weighted trust score
        fusion_result = self.fusion.compute(
            original_text,
            found_articles,
            detectors,
            domain_whitelist,
            domain_blacklist,
        )

        semantic_component = max(0.0, min(1.0, semantic_similarity))
        if "fact-verified" in fact_check_result.flags:
            semantic_component = min(1.0, semantic_component + 0.1)
        if "debunked" in fact_check_result.flags:
            semantic_component = max(0.0, semantic_component - 0.35)

        domain_component = 1.0 if in_whitelist else 0.35
        domain_component += max(0.0, authority_result.score_delta / 120.0)
        domain_component += max(0.0, cross_authority_result.score_delta / 120.0)
        if in_blacklist:
            domain_component -= 0.35
        if "typosquatting" in domain_similarity_result.flags:
            domain_component -= 0.2
        domain_component = max(0.0, min(1.0, domain_component))

        temporal_component = 0.9
        if temporal_result.score_delta < 0:
            temporal_component += temporal_result.score_delta / 40.0
        if "old-news-reposted" in temporal_result.flags:
            temporal_component = 0.2
        elif "old-news" in temporal_result.flags:
            temporal_component = min(temporal_component, 0.6)
        temporal_component = max(0.0, min(1.0, temporal_component))

        ai_penalty = abs(ai_result.score_delta) / 50.0 if ai_result.score_delta < 0 else 0.0
        clickbait_penalty = abs(clickbait_result.score_delta) / 60.0 if clickbait_result.score_delta < 0 else 0.0
        translation_penalty = abs(translation_result.score_delta) / 50.0 if translation_result.score_delta < 0 else 0.0
        linguistic_component = max(0.0, min(1.0, 0.85 - ai_penalty - clickbait_penalty - translation_penalty))

        spam_penalty = 0.0
        for det in [spam_result, duplication_result, single_source_result, reports_result, fake_event_result]:
            if det.score_delta < 0:
                spam_penalty += abs(det.score_delta) / 100.0
        spam_component = max(0.0, min(1.0, 0.85 - spam_penalty))

        donation_component = 1.0
        if donation_result.score_delta < 0:
            donation_component = max(0.0, 1.0 - abs(donation_result.score_delta) / 50.0)
        elif donation_result.score_delta > 0:
            donation_component = min(1.0, 1.0 + donation_result.score_delta / 80.0)

        if cross_authority_result.score_delta > 0:
            linguistic_component = max(linguistic_component, 0.75)
            spam_component = max(spam_component, 0.7)

        weights = {
            "semantic": 0.32,
            "domain": 0.28,
            "temporal": 0.14,
            "linguistic": 0.1,
            "spam": 0.1,
            "donation": 0.06,
        }

        trust_score = 100.0 * (
            weights["semantic"] * semantic_component
            + weights["domain"] * domain_component
            + weights["temporal"] * temporal_component
            + weights["linguistic"] * linguistic_component
            + weights["spam"] * spam_component
            + weights["donation"] * donation_component
        )
        trust_score = _clamp_score(trust_score)

        override = next((d for d in detectors if d.override and d.verdict_override), None)
        if override:
            verdict = override.verdict_override
            override_reason = override.name
        else:
            verdict = self._verdict_from_score(trust_score)
            override_reason = None

        flags: List[str] = []
        evidence: List[str] = []
        for detector in detectors:
            flags.extend(detector.flags)
            evidence.extend(detector.evidence)
        evidence.insert(0, f"semantic similarity {semantic_similarity:.2f}")
        evidence.insert(1, f"authority score {authority_result.score_delta:.0f}")
        if cross_authority_result.score_delta > 0:
            evidence.insert(2, f"cross-source boost {cross_authority_result.score_delta:.0f}")

        confidence_estimate = self._confidence(detectors, trust_score, override is not None)

        summary = "; ".join([ev for ev in evidence[:4]]) or "No strong signals; requires manual review"

        legacy_components = self._components(detectors)
        components = {
            "semantic_similarity": round(semantic_component, 3),
            "domain_reliability": round(domain_component, 3),
            "temporal_consistency": round(temporal_component, 3),
            "linguistic_certainty": round(linguistic_component, 3),
            "spam_behavior_score": round(spam_component, 3),
            "donation_suspicion": round(donation_component, 3),
            "fusion_score": fusion_result["fused_score"],
        }
        components.update({f"legacy_{k}": v for k, v in legacy_components.items()})

        details = {
            "detectors": [asdict(det) for det in detectors],
            "domain_frequency": Counter([a.get("domain") for a in found_articles if a.get("domain")]),
            "meta": meta,
            "fusion": fusion_result,
        }

        return {
            "trust_score": trust_score,
            "verdict": verdict,
            "explanation": summary,
            "flags": sorted(set(flags)),
            "evidence": evidence,
            "confidence_estimate": confidence_estimate,
            "override_reason": override_reason,
            "details": details,
            "components": components,
            "is_donation_post": donation_result.meta.get("is_donation_post", False),
        }

    @staticmethod
    def _verdict_from_score(score: float) -> str:
        if score >= 80:
            return "verified"
        if score >= 60:
            return "needs-review"
        return "likely-false"

    @staticmethod
    def _confidence(detectors: List[DetectorResult], score: float, has_override: bool) -> float:
        positive = len([d for d in detectors if d.score_delta > 0])
        negative = len([d for d in detectors if d.score_delta < 0])
        base = 0.5 + 0.05 * (positive - negative)
        if has_override:
            base = max(base, 0.85)
        if score < 30 or score > 90:
            base += 0.05
        return round(max(0.0, min(1.0, base)), 3)

    @staticmethod
    def _components(detectors: List[DetectorResult]) -> Dict[str, float]:
        def group_score(names: List[str]) -> float:
            total = 100.0
            for det in detectors:
                if det.name in names:
                    total += det.score_delta
            return _clamp_score(total)

        return {
            "content": group_score(["clickbait", "duplication", "translation", "ai-content", "fake-event"]),
            "authority": group_score(["authority", "domain-similarity", "single-source"]),
            "behaviour": group_score(["spam-behavior", "reports"]),
            "safety": group_score(["donation", "phishing", "media-age", "temporal"]),
            "external": group_score(["external-fact-check"]),
        }
