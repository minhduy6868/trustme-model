import sys
from pathlib import Path
import os


# Ensure project src is on path for imports in tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Lighten test env: disable LLM model load and use smaller embedding if needed
os.environ.setdefault("LLM_MODEL_PATH", "")
os.environ.setdefault("SEMANTIC_MODEL_NAME", "sentence-transformers/paraphrase-MiniLM-L3-v2")
