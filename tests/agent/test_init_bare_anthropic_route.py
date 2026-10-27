"""A bare agent routed onto api.anthropic.com binds the native wire, and a native Anthropic init
with no usable token raises the pool's reason instead of building a client that 404s/401s later."""
from types import SimpleNamespace

import pytest


def test_bare_agent_on_anthropic_route_rebinds_to_anthropic_messages(monkeypatch):
    from agent import agent_init

    bound = {}
    monkeypatch.setattr(agent_init, "_routed_client_kwargs",
                        lambda agent, fm, t: {"base_url": "https://api.anthropic.com", "api_key": "sk-ant-x"})
    monkeypatch.setattr(agent_init, "_init_anthropic_client",
                        lambda agent, key, base_url, t: bound.update(key=key, base_url=base_url))
    agent = SimpleNamespace(provider="", requested_provider="", model="claude-x", quiet_mode=True,
                            api_mode="chat_completions")
    agent_init._init_openai_client(agent, None, None, None, 60)
    assert (agent.provider, agent.api_mode) == ("anthropic", "anthropic_messages")
    assert bound == {"key": None, "base_url": "https://api.anthropic.com"}


def test_native_anthropic_without_usable_token_raises(monkeypatch):
    from agent import agent_init, anthropic_credentials
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(anthropic_credentials, "resolve_anthropic_token", lambda *a, **k: None)
    agent = SimpleNamespace(provider="anthropic", model="claude-x", quiet_mode=True)
    with pytest.raises(AuthError):
        agent_init._init_anthropic_client(agent, None, "https://api.anthropic.com", 60)
