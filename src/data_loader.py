"""
Dataset loader and regex compiler for TrustMe model.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

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

    compile_pattern_lists(datasets)
    logger.info("Loaded datasets: %s", ", ".join(datasets.keys()))
    return datasets
