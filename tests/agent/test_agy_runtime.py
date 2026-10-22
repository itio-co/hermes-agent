"""agy runtime (``model.provider: agy``): the stream-json session, its row projection, and the turn wrapper.

A fake ``agy`` replays the event shapes agy 1.4.5 printed in the kido story hermes-agy-agent's probe.
"""

import json
import os
import stat
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import agy_runtime
from agent.transports.agy_session import AgySession

FAKE_AGY = r'''#!__PY__
import json, os, sys, time
log = os.environ["FAKE_AGY_LOG"]
with open(log, "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "pid": os.getpid()}) + "\n")
conv = sys.argv[sys.argv.index("--conversation") + 1] if "--conversation" in sys.argv else "conv-%d" % os.getpid()
def out(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()
def step(i, **kw):
    out({"event": "step_update", "step_update": dict(conversation_id=conv, step_index=i, **kw)})
out({"event": "init", "conversation_id": conv, "init": {"model": "m", "tools": []}})
turns = 0
for line in sys.stdin:
    msg = json.loads(line)
    text = msg["message"]["content"]
    with open(log, "a") as f:
        f.write(json.dumps({"input": text}) + "\n")
    turns += 1
    step(0, state="DONE", step_type="user_input")
    if "DIE" in text:
        sys.exit(3)
    if "SLEEP" in text:
        time.sleep(30)
    if "TOOL" in text:
        step(1, state="ACTIVE", step_type="tool", tool_name="read_file",
             tool_info={"name": "read_file", "parameters": {"Path": "/x/note.txt"}})
        step(1, state="DONE", step_type="tool", tool_name="read_file", duration_seconds=0.01,
             tool_info={"name": "read_file", "parameters": {"Path": "/x/note.txt"}, "output": "1 lines, 17 bytes"})
    if "DENY" in text:
        step(2, state="ACTIVE", step_type="tool", tool_name="run_command",
             tool_info={"name": "run_command", "parameters": {"CommandLine": "echo hi"}})
    step(3, state="ACTIVE", step_type="agent_response", text_delta="done ")
    step(3, state="DONE", step_type="agent_response", text_delta="%d" % turns, usage={"input_tokens": 5})
    result = {"conversation_id": conv, "status": "SUCCESS", "response": "done %d\n" % turns, "num_turns": turns,
              "usage": {"input_tokens": 5, "output_tokens": 1}}
    if "DENY" in text:
        result["denied_actions"] = [{"action": "command", "display_name": "RunCommand"}]
    out({"event": "result", "result": result})
'''


@pytest.fixture
def fake_agy(tmp_path, monkeypatch):
    path = tmp_path / "agy"
    path.write_text(FAKE_AGY.replace("__PY__", sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "agy.log"
    monkeypatch.setenv("FAKE_AGY_LOG", str(log))

    def records(key):
        if not log.exists():
            return []
        return [r[key] for r in map(json.loads, log.read_text().splitlines()) if key in r]
    return SimpleNamespace(bin=str(path), records=records, cwd=str(tmp_path))


def _session(fake, **kw):
    return AgySession(cwd=fake.cwd, model="gemini-test", agy_bin=fake.bin, **kw)


def test_text_turn_projects_one_assistant_row_and_binds_the_conversation(fake_agy):
    session = _session(fake_agy)
    try:
        turn = session.run_turn("hi")
    finally:
        session.close()
    assert turn.error is None and not turn.interrupted
    assert turn.final_text == "done 1"
    assert turn.projected_messages == [{"role": "assistant", "content": "done 1"}]
    assert turn.thread_id and turn.thread_id.startswith("conv-")
    argv = fake_agy.records("argv")[0]
    assert argv[-1] == "-p=" and argv[argv.index("--model") + 1] == "gemini-test"
    assert argv[argv.index("--mode") + 1] == "accept-edits" and "--dangerously-skip-permissions" not in argv


def test_tool_step_becomes_a_tool_call_and_result_pair(fake_agy):
    events = []
    session = _session(fake_agy, on_event=events.append)
    try:
        turn = session.run_turn("TOOL please")
    finally:
        session.close()
    call, result, final = turn.projected_messages
    assert call["tool_calls"][0]["function"]["name"] == "read_file"
    assert json.loads(call["tool_calls"][0]["function"]["arguments"]) == {"path": "/x/note.txt"}
    assert result == {"role": "tool", "tool_call_id": call["tool_calls"][0]["id"], "content": "1 lines, 17 bytes"}
    assert final["content"] == "done 1"
    assert turn.tool_iterations == 1
    assert [e["event"] for e in events].count("result") == 1


def test_second_turn_reuses_the_process(fake_agy):
    session = _session(fake_agy)
    try:
        assert session.run_turn("one").final_text == "done 1"
        assert session.run_turn("two").final_text == "done 2"
    finally:
        session.close()
    assert len(fake_agy.records("argv")) == 1


def test_process_death_retires_the_session_with_an_error(fake_agy):
    session = _session(fake_agy)
    try:
        turn = session.run_turn("DIE")
    finally:
        session.close()
    assert turn.should_retire and "exit 3" in turn.error


def test_interrupt_kills_the_child_and_keeps_the_conversation_id(fake_agy):
    session = _session(fake_agy)
    threading.Timer(0.5, session.request_interrupt).start()
    started = time.monotonic()
    turn = session.run_turn("SLEEP")
    assert time.monotonic() - started < 10
    assert turn.interrupted and turn.should_retire
    assert turn.thread_id and turn.thread_id.startswith("conv-")
    assert session._proc is None


def test_resume_and_permission_flags(fake_agy):
    session = _session(fake_agy, conversation_id="abc", skip_permissions=True)
    try:
        assert session.run_turn("hi").thread_id == "abc"
    finally:
        session.close()
    argv = fake_agy.records("argv")[0]
    assert argv[argv.index("--conversation") + 1] == "abc"
    assert "--dangerously-skip-permissions" in argv and "--mode" not in argv


def test_unfinished_tool_still_gets_a_result_row(fake_agy):
    session = _session(fake_agy)
    try:
        turn = session.run_turn("DENY")
    finally:
        session.close()
    call, result = turn.projected_messages[0], turn.projected_messages[1]
    assert call["tool_calls"][0]["function"]["name"] == "terminal"
    assert result["role"] == "tool" and "did not finish" in result["content"]
    assert turn.denied_actions == [{"action": "command", "display_name": "RunCommand"}]


def test_agy_tools_are_reported_under_hermes_names():
    from agent.transports.agy_session import hermes_tool_view
    assert hermes_tool_view("run_command", {"CommandLine": "ls"},
                            "Command exited with code 2.\r\n\r\nCommand output:\nboom\r\n") == (
        "terminal", {"command": "ls"}, json.dumps({"output": "boom", "exit_code": 2}), True)
    assert hermes_tool_view("edit_file", {"Path": "/a"}, "") == ("patch", {"path": "/a"}, "", False)
    assert hermes_tool_view("search_web", {"query": "q"}, "r") == ("search_web", {"query": "q"}, "r", False)


# -- run_agy_turn -------------------------------------------------------------------------------


def _agent(fake, monkeypatch, *, stored=None):
    monkeypatch.setattr(agy_runtime, "_agy_config", lambda: {"bin": fake.bin})
    db = MagicMock()
    db.get_session_model_config_value.return_value = stored
    agent = MagicMock()
    agent._agy_session = None
    agent._agy_session_started = False
    agent._agy_retired_conversation_id = None
    agent._session_db = db
    agent.session_id = "sess-1"
    agent.session_cwd = fake.cwd
    agent.model = "gemini-test"
    agent._cached_system_prompt = "HERMES PROMPT"
    agent.ephemeral_system_prompt = None
    agent._interrupt_requested = False
    agent._iters_since_skill = 0
    agent._flush_messages_to_session_db.return_value = True
    return agent, db


def _run(agent, text, history=()):
    messages = [*history, {"role": "user", "content": text}]
    return agy_runtime.run_agy_turn(agent, user_message=text, original_user_message=text, messages=messages,
                                    effective_task_id="t")


def test_fresh_conversation_is_seeded_with_the_hermes_prompt_once(fake_agy, monkeypatch):
    agent, db = _agent(fake_agy, monkeypatch)
    try:
        first = _run(agent, "hello", history=[{"role": "user", "content": "earlier"},
                                              {"role": "assistant", "content": "before"}])
        second = _run(agent, "again")
    finally:
        agy_runtime.close_agy_session(agent)
    inputs = fake_agy.records("input")
    assert inputs[0].startswith("<hermes_context>\nHERMES PROMPT") and "earlier" in inputs[0]
    assert inputs[0].endswith("hello") and inputs[1] == "again"
    assert first["completed"] and first["final_response"] == "done 1" and first["agent_persisted"]
    assert second["final_response"] == "done 2"
    session_id, patch = db.patch_session_model_config.call_args[0]
    assert session_id == "sess-1" and patch["agy_conversation_id"].startswith("conv-")


def test_stored_conversation_is_resumed_without_a_seed(fake_agy, monkeypatch):
    agent, _db = _agent(fake_agy, monkeypatch, stored="conv-stored")
    try:
        _run(agent, "hello")
    finally:
        agy_runtime.close_agy_session(agent)
    argv = fake_agy.records("argv")[0]
    assert argv[argv.index("--conversation") + 1] == "conv-stored"
    assert fake_agy.records("input") == ["hello"]


def test_retired_session_respawns_on_the_same_conversation(fake_agy, monkeypatch):
    agent, _db = _agent(fake_agy, monkeypatch)
    try:
        dead = _run(agent, "DIE")
        assert dead["completed"] is False and "exit 3" in dead["error"]
        _run(agent, "next")
    finally:
        agy_runtime.close_agy_session(agent)
    first_argv, second_argv = fake_agy.records("argv")
    assert "--conversation" not in first_argv
    assert second_argv[second_argv.index("--conversation") + 1].startswith("conv-")
    assert fake_agy.records("input")[-1] == "next"


def test_denied_actions_surface_as_a_status_notice(fake_agy, monkeypatch):
    agent, _db = _agent(fake_agy, monkeypatch)
    try:
        _run(agent, "DENY")
    finally:
        agy_runtime.close_agy_session(agent)
    notice = agent._emit_diagnostic_status.call_args[0][0]
    assert "RunCommand" in notice and "approvals.mode: off" in notice


def test_bridge_fires_stream_and_tool_callbacks(fake_agy, monkeypatch):
    agent, _db = _agent(fake_agy, monkeypatch)
    try:
        _run(agent, "TOOL")
    finally:
        agy_runtime.close_agy_session(agent)
    deltas = "".join(c.args[0] for c in agent._fire_stream_delta.call_args_list)
    assert deltas == "done 1"
    start = agent.tool_start_callback.call_args[0]
    complete = agent.tool_complete_callback.call_args[0]
    assert start[0] == complete[0] and start[1] == "read_file" and complete[3] == "1 lines, 17 bytes"


def test_provider_agy_resolves_to_the_agy_api_mode():
    from hermes_cli.runtime_provider import resolve_runtime_provider
    runtime = resolve_runtime_provider(requested="agy")
    assert runtime["provider"] == "agy" and runtime["api_mode"] == "agy"
