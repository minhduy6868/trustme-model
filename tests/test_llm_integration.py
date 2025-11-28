import pytest

from trustscore.llm_adapter import LLMAdapter, LLMFallbackError
from trustscore.analyzers import LanguageRiskScorer
from trustscore.realtime_verifier import RealtimeVerifier
from data_loader import load_datasets


def test_llm_adapter_generate_requires_model_path(monkeypatch):
    monkeypatch.delenv("LLM_MODEL_PATH", raising=False)
    adapter = LLMAdapter()
    with pytest.raises(LLMFallbackError):
        adapter.generate("Hello", timeout=0.1)


def test_language_risk_scorer_combines_llm_score():
    class DummyLLM:
        def classify(self, text, labels, timeout=None):
            return "clickbait"

    scorer = LanguageRiskScorer(llm=DummyLLM())  # type: ignore[arg-type]
    baseline = scorer.score("Bản tin ngắn gọn và trung tính.")
    assert 0.4 < baseline < 0.9  # should combine heuristic with llm penalty


@pytest.mark.asyncio
async def test_realtime_verifier_includes_llm_vectors(monkeypatch, tmp_path):
    class StubLLM:
        logs = []

        def embedding(self, text):
            return [1.0, 0.0, 0.0]

        def classify(self, text, labels, timeout=None):
            return labels[0] if labels else ""

        def generate(self, prompt, timeout=None):
            return '["stub claim"]'

    datasets = load_datasets(tmp_path / "nonexistent")  # empty datasets ok
    engine = RealtimeVerifier(datasets=datasets, llm_adapter=StubLLM())  # type: ignore[arg-type]
    result = await engine.verify("Bản tin thử nghiệm", [], language="vi")

    assert "logs" in result and "llm" in result["logs"]
    assert "V_semantic_llm" in result["components"]
    assert "V_linguistic_llm" in result["components"]
    assert "V_intent_llm" in result["components"]
