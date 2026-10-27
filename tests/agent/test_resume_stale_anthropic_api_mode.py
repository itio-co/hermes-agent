"""A resumed session whose model_config persisted ``api_mode: chat_completions`` on an Anthropic
Messages host heals to the native wire instead of POSTing /chat/completions (404) every turn."""
from types import SimpleNamespace

import pytest


def _agent(base_url, provider=""):
    return SimpleNamespace(provider=provider, model="claude-x", base_url=base_url,
                           _base_url_hostname=base_url.split("://", 1)[-1].split("/", 1)[0],
                           _base_url_lower=base_url.lower())


@pytest.mark.parametrize("provider", ["anthropic", None])
def test_stale_chat_completions_on_anthropic_host_heals(provider):
    from agent.agent_init import _resolve_api_mode

    agent = _agent("https://api.anthropic.com", provider or "")
    _resolve_api_mode(agent, "chat_completions", provider, "https://api.anthropic.com")
    assert (agent.provider, agent.api_mode) == ("anthropic", "anthropic_messages")


def test_explicit_chat_completions_elsewhere_is_kept():
    from agent.agent_init import _resolve_api_mode

    for url, provider in (("https://api.openai.com/v1", "openai"), ("https://openrouter.ai/api/v1", "openrouter")):
        agent = _agent(url, provider)
        _resolve_api_mode(agent, "chat_completions", provider, url)
        assert agent.api_mode == "chat_completions", url
