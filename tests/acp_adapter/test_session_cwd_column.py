"""ACP sessions must populate the cwd COLUMN, not only model_config.

Hermes Desktop's Projects sidebar, ``hermes sessions list``, and every
profile-keyed consumer group sessions off ``sessions.cwd``. The ACP adapter
recorded the workspace only inside the ``model_config`` JSON blob, so every
editor-created session (VS Code, Antigravity, Zed, JetBrains, Buzz) rendered
as unassigned -- "Workspace: --" -- even though its transcript was intact.

``_insert_session_row`` already accepted ``cwd``/``git_repo_root``; the ACP
adapter simply never passed them.
"""
import json
from types import SimpleNamespace

from acp_adapter.session import SessionManager
from hermes_state import SessionDB


def _manager(db):
    return SessionManager(db=db, agent_factory=lambda: SimpleNamespace(model="fixture"))


def test_created_session_records_cwd_in_its_own_column(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    workspace = tmp_path / "hs-wwd"
    workspace.mkdir()
    manager = _manager(db)

    state = manager.create_session(cwd=str(workspace))
    # An empty session stays ephemeral by design (test_empty_session_persistence);
    # content is what mints the row.
    state.history.append({"role": "user", "content": "hello"})
    manager.save_session(state.session_id)

    row = db.get_session(state.session_id)
    assert row["cwd"] == state.cwd, "cwd column must carry the workspace"
    # The JSON copy stays -- _restore() rebuilds the agent from it.
    assert json.loads(row["model_config"])["cwd"] == state.cwd
    db.close()


def test_cwd_is_promoted_when_the_row_already_exists(tmp_path):
    """The create branch is not the live path.

    An agent that owns persistence to this same DB flushes the transcript
    incrementally, so the sessions row is already there by the time the adapter
    persists. ``_persist`` then takes its ``else`` branch, and
    ``update_session_meta`` writes only ``model_config``/``model`` -- leaving the
    column NULL for the entire life of a real editor session.
    """
    db = SessionDB(tmp_path / "state.db")
    workspace = tmp_path / "app-1"
    workspace.mkdir()
    manager = _manager(db)

    state = manager.create_session(cwd=str(workspace))
    state.history.append({"role": "user", "content": "hello"})
    # Stand in for the agent's own incremental flush: the row exists, and it
    # knows nothing about the ACP workspace.
    db.create_session(session_id=state.session_id, source="acp", model="fixture")
    assert db.get_session(state.session_id)["cwd"] in (None, "")

    manager.save_session(state.session_id)

    row = db.get_session(state.session_id)
    assert row["cwd"] == state.cwd, (
        "an existing row must still have its cwd column promoted"
    )
    db.close()


def test_reopening_in_another_workspace_moves_the_cwd_column(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    first, second = tmp_path / "old", tmp_path / "new"
    first.mkdir()
    second.mkdir()
    manager = _manager(db)

    state = manager.create_session(cwd=str(first))
    state.history.append({"role": "user", "content": "hello"})
    manager.save_session(state.session_id)

    manager.update_cwd(state.session_id, str(second))

    row = db.get_session(state.session_id)
    assert row["cwd"] == state.cwd
    assert str(second) in row["cwd"]
    # A moved workspace must claim a fresh probe generation, so a slow git
    # probe for the OLD cwd cannot publish onto the new one.
    assert (row["git_metadata_generation"] or 0) >= 1
    db.close()


def test_legacy_rows_get_their_cwd_column_backfilled_when_the_manager_opens_the_db(tmp_path):
    """A row minted before the adapter wrote the column carries its workspace only in
    ``model_config``; the first DB use by the manager promotes it (#115705)."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="legacy", source="acp", model="m", model_config={"cwd": "/work/hs-wwd"})
    db.create_session(session_id="cli", source="cli", model="m", model_config={"cwd": "/somewhere"})
    assert db.get_session("legacy")["cwd"] in (None, "")

    manager = _manager(db)
    manager.create_session(cwd=str(tmp_path))  # first DB touch

    assert db.get_session("legacy")["cwd"] == "/work/hs-wwd"
    assert db.get_session("cli")["cwd"] in (None, ""), "only ACP rows are repaired"
    db.close()


def test_a_turn_cut_off_before_it_ends_still_leaves_the_cwd(tmp_path):
    """The client process dies mid first turn, so ``_persist`` never runs. The cwd
    recorded at turn start must survive into the row, ``session/list`` and a restore —
    not the ``"."`` default a client would then send back on ``session/load``."""
    db = SessionDB(tmp_path / "state.db")
    workspace = tmp_path / "orbit"
    workspace.mkdir()
    manager = _manager(db)

    state = manager.create_session(cwd=str(workspace))
    manager.persist_cwd(state)  # turn start
    # The agent's own lazy create and incremental flush, which know nothing of the workspace.
    db.create_session(session_id=state.session_id, source="acp", model="fixture",
                      model_config={"reasoning_config": {"enabled": True}})
    db.replace_messages(state.session_id, [{"role": "user", "content": "hello"}])

    row = db.get_session(state.session_id)
    assert row["cwd"] == state.cwd
    assert json.loads(row["model_config"])["cwd"] == state.cwd

    fresh = _manager(db)  # the restarted process
    listed = {s["session_id"]: s for s in fresh.list_sessions()}
    assert listed[state.session_id]["cwd"] == state.cwd
    assert fresh.get_session(state.session_id).cwd == state.cwd
    db.close()


def test_list_and_restore_fall_back_to_the_cwd_column(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="s", source="acp", model="m", model_config={"max_tokens": None}, cwd="/work/orbit")
    db.replace_messages("s", [{"role": "user", "content": "hello"}])
    manager = _manager(db)

    assert {s["session_id"]: s["cwd"] for s in manager.list_sessions()}["s"] == "/work/orbit"
    assert manager.get_session("s").cwd == "/work/orbit"
    db.close()


def test_a_probe_without_a_prompt_stays_ephemeral(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    state = _manager(db).create_session(cwd=str(tmp_path))
    assert db.get_session(state.session_id) is None
    db.close()
