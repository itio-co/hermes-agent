"""Background (async delegate_task) results come back over ACP as a new turn of the dispatching session."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from acp_adapter.server import HermesACPAgent, _Notification
from acp_adapter.session import SessionState
from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def _clean_state():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


class _Conn:
    def __init__(self) -> None:
        self.updates: list = []

    async def session_update(self, session_id, update):
        self.updates.append((session_id, update))


def _event(delegation_id: str, session_key: str = "s1") -> dict:
    event = {
        "type": "async_delegation", "delegation_id": delegation_id, "session_key": session_key,
        "parent_session_id": "h1", "dispatched_at": 1.0, "completed_at": 2.0, "status": "completed",
        "goal": "count the files", "summary": "42 files",
    }
    ad._persist_dispatch(event)
    ad._persist_completion(event, {"status": "completed", "summary": "42 files"})
    return event


def _delivery_state(delegation_id: str) -> str:
    with ad._DB_LOCK, ad._transaction() as conn:
        return conn.execute(
            "SELECT delivery_state FROM async_delegations WHERE delegation_id=?", (delegation_id,)).fetchone()[0]


def _server(state: SessionState | None):
    server = HermesACPAgent.__new__(HermesACPAgent)
    server.session_manager = SimpleNamespace(peek_session=lambda sid: state if state and sid == state.session_id else None)
    server._conn = _Conn()
    prompts: list[str] = []

    async def fake_prompt(*, prompt, session_id, **kwargs):
        prompts.append(prompt[0].text)

    server.prompt = fake_prompt  # type: ignore[method-assign]
    return server, prompts


def _state(*, running: bool = False) -> SessionState:
    return SessionState(session_id="s1", agent=SimpleNamespace(session_id="h1"), is_running=running,
                        runtime_lock=threading.Lock())


def _route(server, evt) -> None:
    async def go():
        server._route_completion(evt, asyncio.get_running_loop())
        await asyncio.sleep(0.05)  # let the scheduled drain run

    asyncio.run(go())


def test_idle_session_runs_the_result_as_a_marked_turn_and_acknowledges_it():
    state = _state()
    server, prompts = _server(state)

    _route(server, _event("deleg_idle"))

    assert len(prompts) == 1 and "deleg_idle" in prompts[0]
    (sid, chunk), = server._conn.updates
    assert sid == "s1" and chunk.session_update == "user_message_chunk"
    note = chunk.field_meta["hermes"]["notification"]
    assert note["kind"] == "async_delegation" and note["delegationId"] == "deleg_idle"
    assert note["title"].startswith("Subagent Task Completed")
    assert _delivery_state("deleg_idle") == "delivered"


def test_busy_session_queues_the_result_behind_the_running_turn():
    state = _state(running=True)
    server, prompts = _server(state)

    _route(server, _event("deleg_busy"))

    assert prompts == [] and len(state.queued_prompts) == 1
    assert isinstance(state.queued_prompts[0], _Notification)
    assert _delivery_state("deleg_busy") == "pending"  # claimed, not yet delivered
    state.is_running = False
    asyncio.run(server._drain_queued_prompts(state, "s1", server._conn))
    assert len(prompts) == 1 and _delivery_state("deleg_busy") == "delivered"


def test_a_result_for_a_session_that_is_not_open_stays_pending():
    server, prompts = _server(_state())

    _route(server, _event("deleg_elsewhere", session_key="other"))

    assert prompts == [] and server._conn.updates == []
    assert _delivery_state("deleg_elsewhere") == "pending"


def test_a_failed_turn_releases_the_claim():
    state = _state()
    server, _ = _server(state)

    async def boom(**kwargs):
        raise RuntimeError("turn failed")

    server.prompt = boom  # type: ignore[method-assign]
    evt = _event("deleg_boom")
    note = _Notification("text")
    note.evt, note.claim, note.label = evt, ad.claim_event_delivery(evt, "acp"), "t"
    state.queued_prompts.append(note)

    with pytest.raises(RuntimeError):
        asyncio.run(server._drain_queued_prompts(state, "s1", server._conn))
    assert _delivery_state("deleg_boom") == "pending"
    assert ad.claim_event_delivery(evt, "acp") is not None, "claim released, so it can be claimed again"


def test_background_count_is_sent_only_when_it_changes(monkeypatch):
    state = _state()
    server, _ = _server(state)
    counts = iter([2, 2, 0])
    monkeypatch.setattr(ad, "live_task_count_for_session", lambda sid: next(counts))

    for _ in range(3):
        asyncio.run(server._send_background_tasks(state))

    sent = [u.field_meta["hermes"]["backgroundTasks"] for _, u in server._conn.updates]
    assert sent == [2, 0]
