from pathlib import Path
from typing import Iterable

import pytest

from data_loader import load_datasets


def _matches(patterns: Iterable, text: str) -> bool:
    return any(p.search(text) for p in patterns)


@pytest.fixture(scope="module")
def datasets():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    return load_datasets(data_dir)


def test_true_samples_not_flagged(datasets):
    samples = datasets["dataset_regression_samples"]
    compiled = datasets["_compiled"]
    vi_patterns = compiled.get("clickbait_patterns_vi", []) + compiled.get("spam_patterns_vi", []) + compiled.get("donation_scam_patterns", [])
    en_patterns = compiled.get("clickbait_patterns_en", []) + compiled.get("spam_patterns_en", []) + compiled.get("donation_scam_patterns", [])

    for text in samples["true_samples_vi"]:
        assert not _matches(vi_patterns, text), f"VI true sample matched blocking pattern: {text}"

    for text in samples["true_samples_en"]:
        assert not _matches(en_patterns, text), f"EN true sample matched blocking pattern: {text}"


def test_fake_samples_trigger_patterns(datasets):
    samples = datasets["dataset_regression_samples"]["fake_samples"]
    compiled = datasets["_compiled"]
    patterns = (
        compiled.get("clickbait_patterns_vi", [])
        + compiled.get("clickbait_patterns_en", [])
        + compiled.get("spam_patterns_vi", [])
        + compiled.get("spam_patterns_en", [])
        + compiled.get("donation_scam_patterns", [])
    )

    for text in samples:
        assert _matches(patterns, text), f"Fake sample not caught by any pattern: {text}"


def test_blacklist_does_not_block_official_sources(datasets):
    blacklist = set(datasets.get("domain_blacklist", []))
    whitelist = set(datasets.get("domain_whitelist", []))
    official_domains = {entry["domain"] for entry in datasets.get("official_sources", [])}
    overlaps = blacklist & (whitelist | official_domains)
    assert not overlaps, f"Blacklist overlaps trusted domains: {overlaps}"
