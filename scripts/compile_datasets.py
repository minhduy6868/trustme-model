#!/usr/bin/env python3
"""
Quick validation/compilation for dataset JSON files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_datasets, compile_pattern_lists  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and compile TrustMe datasets.")
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data"),
        help="Path to dataset directory (default: trustme-model/data)",
    )
    args = parser.parse_args()

    datasets = load_datasets(args.data_dir)
    compiled = datasets.get("_compiled", {})

    print(f"Loaded {len(datasets) - 1 if '_compiled' in datasets else len(datasets)} datasets from {args.data_dir}")
    for name, data in datasets.items():
        if name == "_compiled":
            continue
        size = len(data) if hasattr(data, "__len__") else "n/a"
        print(f" - {name}: {size} entries")

    if not compiled:
        compile_pattern_lists(datasets)
        compiled = datasets.get("_compiled", {})

    print(f"\nCompiled pattern groups: {', '.join(compiled.keys()) or 'none'}")
    for key, patterns in compiled.items():
        print(f" - {key}: {len(patterns)} regex patterns")

    # Write a simple manifest for tooling/debugging
    manifest_path = Path(args.data_dir) / "dataset_manifest.json"
    manifest = {
        "datasets": [k for k in datasets.keys() if k != "_compiled"],
        "compiled_groups": {k: len(v) for k, v in compiled.items()},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
