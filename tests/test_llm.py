import importlib
import sys
import types

import pytest


@pytest.fixture
def real_llm(monkeypatch):
    """Import the real llm.py with a fake OpenAI client (no network, no key)."""
    calls = []
    init_kwargs = {}

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, **kwargs):
            self.base_url = base_url
            init_kwargs.update(kwargs)
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kw: self._create(**kw))
            )

        def _create(self, **kw):
            calls.append(kw)
            msg = types.SimpleNamespace(content="mock answer")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    stub = sys.modules.pop("llm", None)
    mod = importlib.import_module("llm")
    mod.init_kwargs = init_kwargs
    yield mod, calls
    sys.modules.pop("llm", None)
    if stub is not None:
        sys.modules["llm"] = stub


def test_prompt_contains_context_and_question(real_llm):
    llm, calls = real_llm
    assert llm.generate("What is X?", "[Chunk 1]\nX is Y") == "mock answer"
    kw = calls[0]
    user = kw["messages"][1]["content"]
    assert "CONTEXT:\n[Chunk 1]\nX is Y" in user
    assert user.strip().endswith("What is X?")
    assert kw["temperature"] == 0


def test_system_prompt_forbids_outside_knowledge(real_llm):
    llm, calls = real_llm
    llm.generate("q", "")
    system = calls[0]["messages"][0]["content"]
    assert "only the provided context" in system and "Do not invent" in system


def test_non_ascii_survives_prompt(real_llm):
    llm, calls = real_llm
    llm.generate("Что такое ∫?", "контекст 🙂")
    user = calls[0]["messages"][1]["content"]
    assert "Что такое ∫?" in user and "контекст 🙂" in user


def test_client_has_timeout_and_limited_retries(real_llm):
    llm, _ = real_llm
    assert llm.init_kwargs["timeout"] == 30.0
    assert llm.init_kwargs["max_retries"] == 2
