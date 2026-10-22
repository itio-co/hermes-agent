"""agy runtime: hand the whole turn to the Antigravity CLI (``model.provider: agy``).

Mirrors ``run_codex_app_server_turn`` without what agy lacks: no developer-instructions channel (Hermes'
prompt rides in the first message of a fresh agy conversation), no approval requests (headless agy
auto-denies; Hermes' approval bypass maps to ``--dangerously-skip-permissions``), no compaction RPC.
"""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from typing import Any

from agent.codex_runtime import (
    _call_guarded,
    _codex_developer_instructions,
    _consume_user_interrupt,
    _persist_projected_messages,
    _turn_result,
)

logger = logging.getLogger(__name__)

# Durable binding, like codex_thread_id: an agent rebuilt for the same Hermes session resumes agy's conversation.
_AGY_CONVERSATION_KEY = "agy_conversation_id"


def _stored_conversation_id(agent) -> str | None:
    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    if db is None or not session_id:
        return None
    value = db.get_session_model_config_value(session_id, _AGY_CONVERSATION_KEY)
    return value if isinstance(value, str) and value else None


def _store_conversation_id(agent, conversation_id: str | None) -> None:
    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    if db is None or not session_id:
        return
    _call_guarded(db.patch_session_model_config, "agy conversation id could not be stored on the session row",
                  args=(session_id, {_AGY_CONVERSATION_KEY: conversation_id}))


def _user_text(user_message: Any) -> str:
    """agy takes text only; image parts are named so the model knows something was attached."""
    if isinstance(user_message, str):
        return user_message
    if isinstance(user_message, list):
        parts = []
        for part in user_message:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or ("[image attachment not forwarded to agy]"
                                                   if "image" in str(part.get("type")) else ""))
        return "\n".join(p for p in parts if p)
    return str(user_message or "")


def _agy_config() -> dict:
    with suppress(Exception):
        from hermes_cli.config import load_config
        section = load_config().get("agy")
        if isinstance(section, dict):
            return section
    return {}


def make_agy_event_bridge(agent):
    """agy ``step_update`` events -> Hermes' live callbacks (stream deltas, tool cards, progress)."""
    from agent.transports.agy_session import _call_id, _tool_args, hermes_tool_view
    started: dict[Any, tuple[str, dict, float]] = {}

    def cb(attr: str, *args: Any, **kwargs: Any) -> None:
        _call_guarded(getattr(agent, attr, None), f"{attr} raised (agy)", args=args, kwargs=kwargs)

    def on_event(event: dict) -> None:
        if event.get("event") != "step_update":
            return
        step = event.get("step_update") or {}
        index, kind, state = step.get("step_index"), step.get("step_type"), step.get("state")
        cb("_touch_activity", f"agy: {kind} {state}")
        if kind == "agent_response":
            if step.get("text_delta"):
                cb("_fire_stream_delta", step["text_delta"])
            if state == "DONE":
                # Each text step is its own delivered message (see the codex bridge, #74248).
                agent._current_streamed_assistant_text = ""
        elif kind == "tool":
            info = step.get("tool_info") or {}
            name = step.get("tool_name") or info.get("name") or "tool"
            call_id = _call_id(index, name)  # the agy name, like the projected row's id
            if state == "ACTIVE":
                name, args, _out, _err = hermes_tool_view(name, _tool_args(info), "")
                started[index] = (name, args, time.monotonic())
                preview = next((str(v)[:120] for v in args.values() if isinstance(v, str) and v), None)
                cb("tool_progress_callback", "tool.started", name, preview, args)
                cb("tool_start_callback", call_id, name, args)
            elif state == "DONE":
                prior = started.pop(index, None)
                name, args, result, is_error = hermes_tool_view(name, _tool_args(info), str(info.get("output") or ""))
                duration = step.get("duration_seconds")
                if not isinstance(duration, (int, float)):
                    duration = time.monotonic() - prior[2] if prior else None
                cb("tool_progress_callback", "tool.completed", name, None, None,
                   duration=duration, is_error=is_error, result=result)
                cb("tool_complete_callback", call_id, name, args, result)
    return on_event


def close_agy_session(agent) -> None:
    session = getattr(agent, "_agy_session", None)
    agent._agy_session = None
    if session is not None:
        with suppress(Exception):
            session.close()


def _ensure_agy_session(agent) -> bool:
    """Create the session on first use; True when this is a fresh agy conversation (prompt must be seeded)."""
    if getattr(agent, "_agy_session", None) is not None:
        return False
    from agent.runtime_cwd import resolve_agent_cwd
    from agent.transports.agy_session import AgySession
    cfg = _agy_config()
    skip = False
    with suppress(Exception):
        from tools.approval import is_approval_bypass_active
        skip = is_approval_bypass_active()
    # Only an agent's first session resumes the stored conversation; a retired one keeps its own id.
    conversation_id = None if getattr(agent, "_agy_session_started", False) else _stored_conversation_id(agent)
    agent._agy_session_started = True
    agent._agy_session = AgySession(
        cwd=getattr(agent, "session_cwd", None) or str(resolve_agent_cwd()),
        model=getattr(agent, "model", None) or None, agy_bin=cfg.get("bin"),
        conversation_id=conversation_id or getattr(agent, "_agy_retired_conversation_id", None),
        skip_permissions=skip, mode=cfg.get("mode", "accept-edits"),
        on_event=make_agy_event_bridge(agent),
    )
    return agent._agy_session.conversation_id is None


def _seeded_input(agent, user_text: str, messages: list[dict[str, Any]]) -> str:
    """First message of a fresh agy conversation: Hermes' prompt + any prior Hermes turns, then the user text."""
    from agent.codex_runtime_history_seed import render_history_seed
    blocks = [b for b in (_codex_developer_instructions(agent), render_history_seed(messages)) if b]
    if not blocks:
        return user_text
    return "<hermes_context>\n" + "\n\n".join(blocks) + "\n</hermes_context>\n\n" + user_text


def run_agy_turn(agent, *, user_message: Any, original_user_message: Any, messages: list[dict[str, Any]],
                 effective_task_id: str, should_review_memory: bool = False) -> dict[str, Any]:
    """Run one turn on agy and project its steps into ``messages`` (the user row is already there)."""
    fresh = _ensure_agy_session(agent)
    user_text = _user_text(user_message)
    session = agent._agy_session
    try:
        turn = session.run_turn(_seeded_input(agent, user_text, messages) if fresh else user_text)
    except Exception as exc:
        logger.exception("agy turn failed")
        close_agy_session(agent)
        return _turn_result(_consume_user_interrupt(agent), messages, api_calls=0, completed=False, error=str(exc),
                            final_response=f"agy turn failed: {exc}")
    interrupt = _consume_user_interrupt(agent, turn.interrupted)
    if turn.should_retire:
        # Keep the conversation id: the respawned process resumes it with --conversation.
        agent._agy_retired_conversation_id = turn.thread_id
        close_agy_session(agent)
    denied = turn.denied_actions
    if denied:
        names = ", ".join(sorted({str(d.get("display_name") or d.get("action")) for d in denied if isinstance(d, dict)}))
        _call_guarded(getattr(agent, "_emit_diagnostic_status", None), "agy denied-actions notice raised",
                      args=(f"agy blocked {names}: headless agy cannot ask for permission. "
                            "Set approvals.mode: off (or /yolo) on this profile to let it run.",))
    if _persist_projected_messages(agent, turn, messages) and turn.thread_id:
        _store_conversation_id(agent, turn.thread_id)
    agent._iters_since_skill = getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    if not turn.interrupted and turn.error is None:
        _call_guarded(getattr(agent, "_sync_external_memory_for_turn", None), "external memory sync raised", kwargs=dict(
            original_user_message=original_user_message, final_response=turn.final_text, interrupted=False,
            messages=messages))
    final = turn.final_text
    if turn.error and not final:
        final = f"agy: {turn.error}"
    return _turn_result(
        interrupt, messages, api_calls=1, completed=not turn.interrupted and turn.error is None, error=turn.error,
        final_response=final, agent_persisted=True,
    )
