"""Keep the suite offline: heavy / networked modules are replaced before import."""
import sys
import types

import pytest


def _install_stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _FakeModel:
    def encode(self, *a, **k):
        raise AssertionError("embedding model must be mocked in tests")


# embeddings.py loads BAAI/bge-m3 at import; llm.py builds a DeepSeek client.
_install_stub("embeddings", model=_FakeModel(), embed=lambda text: [0.0])
_install_stub("llm", generate=lambda question, context: "stub")


@pytest.fixture
def no_network(monkeypatch):
    import socket

    def blocked(*a, **k):
        raise AssertionError("network access attempted in test")

    monkeypatch.setattr(socket.socket, "connect", blocked)
