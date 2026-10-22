"""One Antigravity CLI (``agy``) print-mode process per AIAgent, driven over stream-json.

``agy --input-format stream-json --output-format stream-json -p=`` reads one
``{"event":"user","message":{"role":"user","content":...}}`` line per turn and writes NDJSON events:
``init``, ``step_update`` (``user_input`` / ``agent_response`` with ``text_delta`` / ``tool`` with
``tool_info``), and a ``result`` that ends the turn. agy has no interrupt message, so an interrupt
kills the process; the next turn respawns it with ``--conversation <id>``, which keeps agy's history.
Shapes were taken from agy 1.4.5 (kido story hermes-agy-agent, ``artifacts/agy-probe.md``).
"""

from __future__ import annotations

import collections
import json
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.transports.codex_app_server_session import TurnResult

logger = logging.getLogger(__name__)


@dataclass
class AgyTurnResult(TurnResult):
    """codex's TurnResult (``thread_id`` = agy conversation id) plus the actions headless agy denied, and
    whether agy produced any step at all (a resume that fails before its first step is a bad conversation id)."""

    denied_actions: list[dict] = field(default_factory=list)
    started: bool = False
    # agy answers an unknown --conversation with a NEW conversation (stderr: 'conversation "<id>" not found').
    resume_lost: bool = False

_STDERR_TAIL_LINES = 20
_INIT_TIMEOUT = 120.0  # agy prints ``init`` before reading stdin; a resume took 33 s in the probe
_POLL_SECONDS = 0.25
# No event at all for this long ends the turn (a wedged agy: stuck auth refresh, network stall).
DEFAULT_TURN_IDLE_TIMEOUT = 1800.0  # long silent tools (builds) emit nothing until DONE
_EOF = object()


def agy_binary(configured: Optional[str] = None) -> Optional[str]:
    """Path of the agy executable: ``configured`` (``agy.bin``), else ``agy`` on PATH."""
    return shutil.which(configured or "agy")


def _call_id(turn_tag: str, step_index: Any, name: str) -> str:
    """Unique per turn: agy's step_index may restart per process, and duplicate ids break providers
    that the session might later switch to (Anthropic rejects repeated tool_use ids)."""
    return f"agy_{name or 'tool'}_{turn_tag}_{step_index}"


def _error_text(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or json.dumps(error, ensure_ascii=False))
    return str(error)


def _tool_args(info: dict) -> dict:
    params = info.get("parameters")
    return params if isinstance(params, dict) else ({"arguments": params} if params is not None else {})


# agy tool -> (Hermes tool, {agy param: Hermes param}). Hermes' names get its tool cards and result
# formatting (ACP titles, terminal output, edit rows) and keep the transcript readable by other runtimes.
_HERMES_TOOL_NAMES = {
    "run_command": ("terminal", {"CommandLine": "command", "Cwd": "workdir"}),
    "read_file": ("read_file", {"Path": "path", "AbsolutePath": "path"}),
    "view_file": ("read_file", {"Path": "path", "AbsolutePath": "path"}),
    "write_file": ("write_file", {"Path": "path", "TargetFile": "path"}),
    "write_to_file": ("write_file", {"Path": "path", "TargetFile": "path"}),
    "edit_file": ("patch", {"Path": "path", "TargetFile": "path"}),
    "replace_file_content": ("patch", {"Path": "path", "TargetFile": "path"}),
    "multi_replace_file_content": ("patch", {"Path": "path", "TargetFile": "path"}),
}
_COMMAND_RESULT = re.compile(r"^Command exited with code (-?\d+)\.(?:\s*Command output:\n(.*)|.*)$", re.DOTALL)


def hermes_tool_view(name: str, args: dict, output: str) -> tuple[str, dict, str, bool]:
    """(tool name, args, result text, is_error) as Hermes' own tool would have reported them."""
    output = output.replace("\r\n", "\n")
    mapped = _HERMES_TOOL_NAMES.get(name)
    if mapped is None:
        return name, args, output, False
    hermes_name, keys = mapped
    hermes_args = {keys.get(k, k): v for k, v in args.items()}
    if hermes_name == "terminal" and (m := _COMMAND_RESULT.match(output.strip())):
        exit_code = int(m.group(1))
        return hermes_name, hermes_args, json.dumps({"output": (m.group(2) or "").rstrip(), "exit_code": exit_code},
                                                    ensure_ascii=False), exit_code != 0
    return hermes_name, hermes_args, output, False


class AgySession:
    """Owns the agy child; ``run_turn`` is called from one worker thread at a time."""

    def __init__(
        self,
        *,
        cwd: str,
        model: Optional[str] = None,
        agy_bin: Optional[str] = None,
        conversation_id: Optional[str] = None,
        skip_permissions: bool = False,
        mode: Optional[str] = "accept-edits",
        on_event: Optional[Callable[[dict], None]] = None,
        idle_timeout: float = DEFAULT_TURN_IDLE_TIMEOUT,
    ) -> None:
        self.cwd, self.model, self.agy_bin = cwd, model, agy_bin
        self.conversation_id = conversation_id
        # True between a spawn whose init named a conversation other than the one asked for (or none was
        # asked for) and that process's first message: the caller's seed must go with it.
        self._fresh_conversation = False
        self._resume_lost = False
        self.skip_permissions, self.mode, self.on_event = skip_permissions, mode, on_event
        self.idle_timeout = idle_timeout
        self._proc: Optional[subprocess.Popen] = None
        self._events: queue.Queue[Any] = queue.Queue()
        self._stderr: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)
        self._interrupt = threading.Event()

    def spawn_settings(self) -> tuple:
        """What the child was started with; a change needs a respawn (flags are fixed per process)."""
        return (self.model, self.skip_permissions, self.mode, self.cwd, self.agy_bin)

    # -- process -------------------------------------------------------------------------------

    def argv(self) -> list[str]:
        binary = agy_binary(self.agy_bin)
        if binary is None:
            raise RuntimeError(f"agy executable not found ({self.agy_bin or 'agy'}); install the Antigravity CLI")
        argv = [binary, "--input-format", "stream-json", "--output-format", "stream-json"]
        if self.model:
            argv += ["--model", self.model]
        if self.conversation_id:
            argv += ["--conversation", self.conversation_id]
        if self.skip_permissions:
            argv.append("--dangerously-skip-permissions")
        elif self.mode:
            argv += ["--mode", self.mode]
        # ``-p`` takes the next argument as its prompt, so it goes last with an empty inline value.
        return argv + ["-p="]

    def _spawn(self) -> subprocess.Popen:
        # Per-process queue and stderr tail: a killed child's reader threads keep writing to their own.
        self._events, self._stderr = queue.Queue(), collections.deque(maxlen=_STDERR_TAIL_LINES)
        proc = subprocess.Popen(
            self.argv(), cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, start_new_session=True,
            env={**os.environ, "NO_COLOR": "1"},
        )
        threading.Thread(target=self._read_stdout, args=(proc, self._events), daemon=True, name="agy-stdout").start()
        threading.Thread(target=self._read_stderr, args=(proc, self._stderr), daemon=True, name="agy-stderr").start()
        self._proc = proc
        return proc

    @staticmethod
    def _read_stdout(proc: subprocess.Popen, events: queue.Queue[Any]) -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.strip()
                if not line:
                    continue
                try:
                    events.put(json.loads(line))
                except ValueError:
                    logger.debug("agy: non-JSON stdout line: %s", line[:200])
        except (OSError, ValueError):
            pass
        finally:
            events.put(_EOF)

    @staticmethod
    def _read_stderr(proc: subprocess.Popen, tail: collections.deque[str]) -> None:
        try:
            for line in proc.stderr:  # type: ignore[union-attr]
                if line.strip():
                    tail.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        # Signal the whole group even if the leader already exited: run_command shells live in it.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            with suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            with suppress(Exception):
                pipe.close()  # type: ignore[union-attr]

    def request_interrupt(self) -> None:
        self._interrupt.set()

    # -- turn ----------------------------------------------------------------------------------

    def _await_init(self, proc: subprocess.Popen) -> Optional[str]:
        """Read the ``init`` agy prints at start-up; returns an error text when it never comes."""
        requested = self.conversation_id
        deadline = time.monotonic() + _INIT_TIMEOUT
        while time.monotonic() < deadline:
            if self._interrupt.is_set():
                return None
            try:
                event = self._events.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                continue
            if event is _EOF:
                return f"agy exited at start-up (exit {proc.poll()}). {self.stderr_tail()}".strip()
            if isinstance(event, dict) and event.get("event") == "init":
                self.conversation_id = event.get("conversation_id") or None
                self._resume_lost = bool(requested) and self.conversation_id != requested
                self._fresh_conversation = not requested or self._resume_lost
                return None
        return f"agy printed no init within {_INIT_TIMEOUT:.0f}s. {self.stderr_tail()}".strip()

    def run_turn(self, user_input: str, fresh_input: Optional[str] = None) -> AgyTurnResult:
        """``fresh_input`` replaces ``user_input`` when this turn opens a new agy conversation (Hermes' seed)."""
        result = AgyTurnResult(submitted_user_text=user_input)
        try:
            self._run_turn(user_input, fresh_input, result)
        finally:
            # Cleared only at the end, so a stop that lands while spawning or writing still counts.
            self._interrupt.clear()
        result.thread_id = self.conversation_id
        return result

    def _run_turn(self, user_input: str, fresh_input: Optional[str], result: AgyTurnResult) -> None:
        if self._interrupt.is_set():
            result.interrupted = True
            return
        proc = self._proc
        if proc is None or proc.poll() is not None:
            self.close()
            proc = self._spawn()
            if error := self._await_init(proc):
                result.error, result.should_retire = error, True
                self.close()
                return
            result.resume_lost = self._resume_lost
        if self._fresh_conversation and fresh_input is not None:
            user_input = fresh_input
        events = self._events
        # Anything left after the previous turn's result belongs to no turn.
        with suppress(queue.Empty):
            while True:
                if events.get_nowait() is _EOF:
                    events.put(_EOF)
                    break
        if self._interrupt.is_set():
            result.interrupted = True
            return
        line = json.dumps({"event": "user", "message": {"role": "user", "content": user_input}}, ensure_ascii=False)
        try:
            proc.stdin.write(line + "\n")  # type: ignore[union-attr]
            proc.stdin.flush()  # type: ignore[union-attr]
            self._fresh_conversation = False
        except (BrokenPipeError, OSError, ValueError) as exc:
            result.error, result.should_retire = f"agy exited before the turn started: {exc}. {self.stderr_tail()}".strip(), True
            return
        turn_tag = uuid.uuid4().hex[:8]
        projector = _Projector(turn_tag)
        last_event = time.monotonic()
        while True:
            if self._interrupt.is_set():
                result.interrupted = result.should_retire = True
                self.close()
                break
            try:
                event = events.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                if self.idle_timeout and time.monotonic() - last_event > self.idle_timeout:
                    result.error = f"agy sent nothing for {self.idle_timeout:.0f}s; the turn was stopped. {self.stderr_tail()}".strip()
                    result.should_retire = True
                    self.close()
                    break
                continue
            last_event = time.monotonic()
            if event is _EOF:
                result.error = f"agy exited mid-turn (exit {proc.poll()}). {self.stderr_tail()}".strip()
                result.should_retire = True
                break
            if not isinstance(event, dict):
                continue
            if event.get("event") == "init":
                self.conversation_id = event.get("conversation_id") or self.conversation_id
                continue
            event["_turn"] = turn_tag
            if self.on_event is not None:
                try:
                    self.on_event(event)
                except Exception:
                    logger.debug("agy on_event raised", exc_info=True)
            if event.get("event") == "step_update":
                step = event.get("step_update") or {}
                result.started = True
                self.conversation_id = step.get("conversation_id") or self.conversation_id
                projector.step(step)
            elif event.get("event") == "result":
                final = event.get("result") or {}
                self.conversation_id = final.get("conversation_id") or self.conversation_id
                projector.finish(final)
                result.final_text = (final.get("response") or projector.last_text or "").strip()
                if str(final.get("status") or "").upper() != "SUCCESS":
                    detail = _error_text(final.get("error") or f"agy turn ended with status {final.get('status')}")
                    result.error = f"{detail} {self.stderr_tail()}".strip()
                usage = final.get("usage")
                if isinstance(usage, dict):
                    result.token_usage_last = usage
                result.denied_actions = list(final.get("denied_actions") or [])
                break
        result.projected_messages = projector.messages
        result.tool_iterations = projector.tool_iterations


class _Projector:
    """``step_update`` events -> OpenAI-shaped rows: each agent text step becomes an assistant row,
    each finished tool step an assistant tool_call + tool result pair (the codex projector's shape)."""

    def __init__(self, turn_tag: str = "") -> None:
        self.turn_tag = turn_tag
        self.messages: list[dict] = []
        self.tool_iterations = 0
        self.last_text = ""
        self._text: dict[Any, list[str]] = {}
        self._tools: dict[Any, dict] = {}
        self._flushed: set = set()

    def step(self, step: dict) -> None:
        index, kind, state = step.get("step_index"), step.get("step_type"), step.get("state")
        if kind == "agent_response":
            if step.get("text_delta"):
                self._text.setdefault(index, []).append(step["text_delta"])
            if state == "DONE":
                # A tool step that never finished (denied) belongs before the text that followed it.
                self._flush_tools(before=index)
                text = "".join(self._text.pop(index, [])).strip()
                if text:
                    self.last_text = text
                    self.messages.append({"role": "assistant", "content": text})
        elif kind == "tool":
            info = step.get("tool_info") or {}
            name = step.get("tool_name") or info.get("name") or "tool"
            if index in self._flushed:
                return  # already closed with a "did not finish" row; a late DONE must not repeat the id
            if state == "ACTIVE":
                self._tools[index] = {"name": name, "args": _tool_args(info)}
            elif state == "DONE":
                started = self._tools.pop(index, None) or {"name": name, "args": _tool_args(info)}
                self._emit_tool(index, started["name"], started["args"], str(info.get("output") or ""))

    def _emit_tool(self, index: Any, name: str, args: dict, output: str) -> None:
        call_id = _call_id(self.turn_tag, index, name)
        name, args, output, _is_error = hermes_tool_view(name, args, output)
        self.messages.append({"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False, sort_keys=True)},
        }]})
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
        self.tool_iterations += 1

    def _flush_tools(self, before: Any = None) -> None:
        """A tool that never reported DONE (denied, cancelled) still needs its result row."""
        for index in sorted(self._tools, key=lambda i: (not isinstance(i, int), i if isinstance(i, int) else 0)):
            if before is not None and isinstance(index, int) and isinstance(before, int) and index >= before:
                continue
            started = self._tools.pop(index)
            self._flushed.add(index)
            self._emit_tool(index, started["name"], started["args"], "[no result: tool did not finish]")

    def finish(self, final: dict) -> None:
        self._flush_tools()
        for index in list(self._text):
            text = "".join(self._text.pop(index)).strip()
            if text:
                self.last_text = text
                self.messages.append({"role": "assistant", "content": text})
        response = (final.get("response") or "").strip()
        if response and response != self.last_text:
            self.last_text = response
            self.messages.append({"role": "assistant", "content": response})
