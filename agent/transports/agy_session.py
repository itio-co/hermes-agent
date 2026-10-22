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
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.transports.codex_app_server_session import TurnResult

logger = logging.getLogger(__name__)


@dataclass
class AgyTurnResult(TurnResult):
    """codex's TurnResult (``thread_id`` = agy conversation id) plus the actions headless agy denied."""

    denied_actions: list[dict] = field(default_factory=list)

_STDERR_TAIL_LINES = 20
_POLL_SECONDS = 0.25
_EOF = object()


def agy_binary(configured: Optional[str] = None) -> Optional[str]:
    """Path of the agy executable: ``configured`` (``agy.bin``), else ``agy`` on PATH."""
    return shutil.which(configured or "agy")


def _call_id(step_index: Any, name: str) -> str:
    return f"agy_{name or 'tool'}_{step_index}"


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
    ) -> None:
        self.cwd, self.model, self.agy_bin = cwd, model, agy_bin
        self.conversation_id = conversation_id
        self.skip_permissions, self.mode, self.on_event = skip_permissions, mode, on_event
        self._proc: Optional[subprocess.Popen] = None
        self._events: queue.Queue[Any] = queue.Queue()
        self._stderr: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)
        self._interrupt = threading.Event()

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

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _spawn(self) -> None:
        self._events = queue.Queue()
        self._stderr.clear()
        self._proc = subprocess.Popen(
            self.argv(), cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, start_new_session=True,
            env={**os.environ, "NO_COLOR": "1"},
        )
        threading.Thread(target=self._read_stdout, args=(self._proc, self._events), daemon=True,
                         name="agy-stdout").start()
        threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True, name="agy-stderr").start()

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
        finally:
            events.put(_EOF)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:  # type: ignore[union-attr]
            if line.strip():
                self._stderr.append(line.rstrip())

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                logger.debug("agy: kill failed", exc_info=True)

    def request_interrupt(self) -> None:
        self._interrupt.set()

    # -- turn ----------------------------------------------------------------------------------

    def run_turn(self, user_input: str) -> AgyTurnResult:
        result = AgyTurnResult(submitted_user_text=user_input)
        self._interrupt.clear()
        if not self._alive():
            self._spawn()
        line = json.dumps({"event": "user", "message": {"role": "user", "content": user_input}}, ensure_ascii=False)
        try:
            self._proc.stdin.write(line + "\n")  # type: ignore[union-attr]
            self._proc.stdin.flush()  # type: ignore[union-attr]
        except (BrokenPipeError, OSError) as exc:
            result.error, result.should_retire = f"agy exited before the turn started: {exc}. {self.stderr_tail()}".strip(), True
            return result
        projector = _Projector()
        while True:
            if self._interrupt.is_set():
                result.interrupted = result.should_retire = True
                self.close()
                break
            try:
                event = self._events.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                continue
            if event is _EOF:
                code = self._proc.poll() if self._proc else None
                result.error = f"agy exited mid-turn (exit {code}). {self.stderr_tail()}".strip()
                result.should_retire = True
                break
            if not isinstance(event, dict):
                continue
            if event.get("event") == "init":
                self.conversation_id = event.get("conversation_id") or self.conversation_id
                continue
            if self.on_event is not None:
                try:
                    self.on_event(event)
                except Exception:
                    logger.debug("agy on_event raised", exc_info=True)
            if event.get("event") == "step_update":
                step = event.get("step_update") or {}
                self.conversation_id = step.get("conversation_id") or self.conversation_id
                projector.step(step)
            elif event.get("event") == "result":
                final = event.get("result") or {}
                self.conversation_id = final.get("conversation_id") or self.conversation_id
                projector.finish(final)
                result.final_text = (final.get("response") or projector.last_text or "").strip()
                if str(final.get("status") or "").upper() != "SUCCESS":
                    result.error = final.get("error") or f"agy turn ended with status {final.get('status')}"
                usage = final.get("usage")
                if isinstance(usage, dict):
                    result.token_usage_last = usage
                result.denied_actions = list(final.get("denied_actions") or [])
                break
        result.projected_messages = projector.messages
        result.tool_iterations = projector.tool_iterations
        result.thread_id = self.conversation_id
        return result


class _Projector:
    """``step_update`` events -> OpenAI-shaped rows: each agent text step becomes an assistant row,
    each finished tool step an assistant tool_call + tool result pair (the codex projector's shape)."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.tool_iterations = 0
        self.last_text = ""
        self._text: dict[Any, list[str]] = {}
        self._tools: dict[Any, dict] = {}

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
            if state == "ACTIVE":
                self._tools[index] = {"name": name, "args": _tool_args(info)}
            elif state == "DONE":
                started = self._tools.pop(index, None) or {"name": name, "args": _tool_args(info)}
                self._emit_tool(index, started["name"], started["args"], str(info.get("output") or ""))

    def _emit_tool(self, index: Any, name: str, args: dict, output: str) -> None:
        call_id = _call_id(index, name)
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
