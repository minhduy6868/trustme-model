"""
Dataset loader and regex compiler for TrustMe model.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)


def load_json(path: Path) -> Any:
    """Load a JSON file with UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _compile_patterns(value: Any) -> list[re.Pattern]:
    """Compile regex patterns from a list of strings or dicts with 'pattern'."""
    compiled: list[re.Pattern] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                try:
                    compiled.append(re.compile(item, re.I | re.U))
                except re.error as exc:
                    logger.warning("Skip invalid pattern %s: %s", item, exc)
            elif isinstance(item, dict) and isinstance(item.get("pattern"), str):
                try:
                    compiled.append(re.compile(item["pattern"], re.I | re.U))
                except re.error as exc:
                    logger.warning("Skip invalid pattern %s: %s", item.get("pattern"), exc)
    return compiled


def compile_pattern_lists(datasets: Dict[str, Any]) -> Dict[str, list[re.Pattern]]:
    """
    Compile regex pattern lists (case-insensitive, unicode).
    Stores compiled patterns under datasets['_compiled'].
    """
    compiled: Dict[str, list[re.Pattern]] = {}
    for key, value in datasets.items():
        patterns = _compile_patterns(value)
        if patterns:
            compiled[key] = patterns
    datasets["_compiled"] = compiled
    return compiled


def _make_key(base: Path, path: Path) -> str:
    rel = path.relative_to(base)
    rel_no_suffix = rel.with_suffix("")
    return rel_no_suffix.as_posix().replace("/", "__")


def _dedup_list(values: Iterable[Any]) -> List[Any]:
    seen = set()
    output: List[Any] = []
    for val in values:
        key = json.dumps(val, sort_keys=True) if isinstance(val, (dict, list)) else val
        if key in seen:
            continue
        seen.add(key)
        output.append(val)
    return output


def _normalize_trusted_sources(datasets: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]], List[str], int]:
    merged_from: List[str] = []
    trusted_sources: List[Dict[str, Any]] = []

    existing = datasets.get("trusted_sources", [])
    if isinstance(existing, list):
        trusted_sources.extend(existing)
        merged_from.append("trusted_sources")

    # Merge domain whitelist + official sources into trusted_sources
    domain_whitelist = datasets.get("domain_whitelist", [])
    if isinstance(domain_whitelist, list) and domain_whitelist:
        merged_from.append("domain_whitelist")
        trusted_sources.extend(
            [
                {"name": d, "type": "domain", "identifier": d, "trust_score": 0.88}
                for d in domain_whitelist
            ]
        )

    official_sources = datasets.get("official_sources", [])
    if isinstance(official_sources, list) and official_sources:
        merged_from.append("official_sources")
        for entry in official_sources:
            domain = (entry.get("domain") or "").lower()
            if not domain:
                continue
            trusted_sources.append(
                {
                    "name": entry.get("name") or domain,
                    "type": "domain",
                    "identifier": domain,
                    "trust_score": float(entry.get("reputation", 0.9)),
                }
            )

    # Deduplicate by identifier
    merged: Dict[str, Dict[str, Any]] = {}
    for item in trusted_sources:
        ident = (item.get("identifier") or "").lower()
        if not ident:
            continue
        current = merged.get(ident)
        if not current or item.get("trust_score", 0) > current.get("trust_score", 0):
            merged[ident] = item

    normalized = list(merged.values())
    domains = sorted({i["identifier"] for i in normalized if i.get("type") == "domain"})
    pages = len([i for i in normalized if i.get("type") == "page"])
    datasets["trusted_sources"] = normalized
    datasets["domain_whitelist"] = domains  # keep compatibility but now sourced from trusted_sources
    return domains, normalized, merged_from, pages


def _normalize_text_quality(datasets: Dict[str, Any]) -> Dict[str, Any]:
    merged_from: List[str] = []
    clickbait_vi = datasets.pop("clickbait_patterns_vi", []) or []
    clickbait_en = datasets.pop("clickbait_patterns_en", []) or []
    spam_vi = datasets.pop("spam_patterns_vi", []) or []
    spam_en = datasets.pop("spam_patterns_en", []) or []
    spam_indicators = datasets.pop("spam_indicators", []) or []

    if clickbait_vi or clickbait_en or spam_vi or spam_en or spam_indicators:
        merged_from.extend(
            [
                name
                for name, values in [
                    ("clickbait_patterns_vi", clickbait_vi),
                    ("clickbait_patterns_en", clickbait_en),
                    ("spam_patterns_vi", spam_vi),
                    ("spam_patterns_en", spam_en),
                    ("spam_indicators", spam_indicators),
                ]
                if values
            ]
        )

    quality = datasets.get("text_quality_patterns", {}) or {}
    clickbait_group = quality.get("clickbait", {}) if isinstance(quality, dict) else {}
    spam_group = quality.get("spam", {}) if isinstance(quality, dict) else {}
    indicators_group = quality.get("spam_indicators", []) if isinstance(quality, dict) else []

    merged_clickbait_vi = _dedup_list((clickbait_group.get("vi") or []) + clickbait_vi)
    merged_clickbait_en = _dedup_list((clickbait_group.get("en") or []) + clickbait_en)
    merged_spam_vi = _dedup_list((spam_group.get("vi") or []) + spam_vi)
    merged_spam_en = _dedup_list((spam_group.get("en") or []) + spam_en)
    merged_spam_indicators = _dedup_list(indicators_group + spam_indicators)

    datasets["text_quality_patterns"] = {
        "clickbait": {"vi": merged_clickbait_vi, "en": merged_clickbait_en},
        "spam": {"vi": merged_spam_vi, "en": merged_spam_en},
        "spam_indicators": merged_spam_indicators,
    }
    # Compatibility keys used across analyzers
    datasets["clickbait_patterns_vi"] = merged_clickbait_vi
    datasets["clickbait_patterns_en"] = merged_clickbait_en
    datasets["spam_patterns_vi"] = merged_spam_vi
    datasets["spam_patterns_en"] = merged_spam_en
    datasets["spam_indicators"] = merged_spam_indicators

    removed = [k for k in merged_from if k in datasets]
    for k in removed:
        datasets.pop(k, None)

    logger.info(
        "Merged text quality datasets from %s into text_quality_patterns (vi:%d/en:%d spam_vi:%d spam_en:%d)",
        merged_from or ["text_quality_patterns only"],
        len(merged_clickbait_vi),
        len(merged_clickbait_en),
        len(merged_spam_vi),
        len(merged_spam_en),
    )
    return datasets


def _normalize_donation(datasets: Dict[str, Any]) -> None:
    merged_from: List[str] = []
    donation_data = datasets.get("donation_patterns", {}) or {}

    indicators = donation_data.get("indicators", []) if isinstance(donation_data, dict) else []
    bank_patterns = donation_data.get("bank_account_patterns", []) if isinstance(donation_data, dict) else []
    scam_patterns = donation_data.get("scam_patterns", []) if isinstance(donation_data, dict) else []
    official_accounts = donation_data.get("official_accounts", {}) if isinstance(donation_data, dict) else {}
    # Merge external official accounts dataset if provided
    extra_official = datasets.get("donation_official_accounts", {})
    if isinstance(extra_official, dict) and extra_official:
        merged_from.append("donation_official_accounts")
        for org, accounts in extra_official.items():
            current = official_accounts.get(org, [])
            if isinstance(accounts, list):
                current.extend(accounts)
                official_accounts[org] = current

    for key, store in [
        ("donation_indicators", indicators),
        ("bank_account_patterns", bank_patterns),
        ("donation_scam_patterns", scam_patterns),
    ]:
        if datasets.get(key):
            merged_from.append(key)
            incoming = datasets.pop(key) or []
            if isinstance(store, list):
                store.extend(incoming)
        elif store:
            merged_from.append("donation_patterns")

    if datasets.get("official_accounts"):
        merged_from.append("official_accounts")
        if isinstance(official_accounts, dict):
            # Merge official account dict shallowly
            for org, accounts in datasets["official_accounts"].items():
                existing = official_accounts.get(org, [])
                official_accounts[org] = _dedup_list(existing + (accounts or []))

    datasets["donation_patterns"] = {
        "indicators": _dedup_list(indicators),
        "bank_account_patterns": _dedup_list(bank_patterns),
        "scam_patterns": _dedup_list(scam_patterns),
        "official_accounts": official_accounts,
    }
    datasets["donation_indicators"] = datasets["donation_patterns"]["indicators"]
    datasets["bank_account_patterns"] = datasets["donation_patterns"]["bank_account_patterns"]
    datasets["donation_scam_patterns"] = datasets["donation_patterns"]["scam_patterns"]
    datasets["official_accounts"] = datasets["donation_patterns"]["official_accounts"]

    logger.info(
        "Merged donation datasets from %s into donation_patterns (indicators:%d, bank_patterns:%d, scam:%d, orgs:%d)",
        merged_from or ["donation_patterns only"],
        len(datasets['donation_patterns']['indicators']),
        len(datasets['donation_patterns']['bank_account_patterns']),
        len(datasets['donation_patterns']['scam_patterns']),
        len(datasets['donation_patterns']['official_accounts']),
    )


def load_datasets(data_dir: str | os.PathLike[str]) -> Dict[str, Any]:
    """
    Load all JSON files in data_dir into a dict keyed by stem.
    Compiles regex lists for quick reuse.
    """
    data_path = Path(data_dir)
    datasets: Dict[str, Any] = {}

    if not data_path.exists():
        logger.warning("Data directory %s does not exist; using empty datasets", data_path)
        datasets["_compiled"] = {}
        return datasets

    for fname in sorted(data_path.rglob("*.json")):
        key = _make_key(data_path, fname)
        try:
            datasets[key] = load_json(fname)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load dataset %s: %s", fname, exc)

    # Normalize merged datasets and log sources
    domains, trusted_sources, merged_from, page_count = _normalize_trusted_sources(datasets)
    _normalize_text_quality(datasets)
    _normalize_donation(datasets)

    compile_pattern_lists(datasets)

    logger.info(
        "Loaded datasets (%d files). Trusted sources merged from %s (domains: %d, total entries: %d)",
        len(list(data_path.rglob('*.json'))),
        merged_from or ["trusted_sources only"],
        len(domains),
        len(trusted_sources),
    )
    logger.info("Trusted sources breakdown: %d domains, %d pages", len(domains), page_count)
    return datasets
