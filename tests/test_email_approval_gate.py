"""
Prove that send_email and reply_to_email cannot reach SMTP when the
agent_email_confirm gate is on (the default).
"""
import os
import sys
import tempfile
import sqlite3
import types
from pathlib import Path

_tmp_data = Path(tempfile.mkdtemp(prefix="odysseus-approval-gate-test-"))
os.environ["ODYSSEUS_DATA_DIR"] = str(_tmp_data)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_data / 'app.db'}")

# Stub out the mcp package so email_server can import
_mcp_stub = types.ModuleType("mcp")
_mcp_server_stub = types.ModuleType("mcp.server")
_mcp_stdio_stub = types.ModuleType("mcp.server.stdio")
_mcp_types_stub = types.ModuleType("mcp.types")

class _FakeServer:
    def __init__(self, name): pass
    def list_tools(self): return lambda fn: fn
    def call_tool(self): return lambda fn: fn

_mcp_server_stub.Server = _FakeServer
_mcp_stdio_stub.stdio_server = None
_mcp_types_stub.Tool = type("Tool", (), {"__init__": lambda *a, **kw: None})
_mcp_types_stub.TextContent = type("TextContent", (), {"__init__": lambda self, **kw: [setattr(self, k, v) for k, v in kw.items()]})

sys.modules.setdefault("mcp", _mcp_stub)
sys.modules.setdefault("mcp.server", _mcp_server_stub)
sys.modules.setdefault("mcp.server.stdio", _mcp_stdio_stub)
sys.modules.setdefault("mcp.types", _mcp_types_stub)

import mcp_servers.email_server as es


def _smtp_should_not_be_called(*a, **kw):
    raise AssertionError("SMTP connection attempted — approval gate failed!")


def test_send_email_stashes_draft_when_gate_on(monkeypatch):
    """send_email must NOT call _smtp_connect when agent_email_confirm is True."""
    monkeypatch.setattr(es, "_read_agent_email_confirm_setting", lambda: True)
    monkeypatch.setattr(es, "_smtp_connect", _smtp_should_not_be_called)
    monkeypatch.setattr(es, "_load_config", lambda acct=None: {
        "account_id": "test", "account_name": "test",
        "from_address": "test@example.com",
    })

    result = es._send_email(
        to="victim@example.com",
        subject="hallucinated email",
        body="the model made this up",
    )

    assert result.get("pending") is True, "Draft was not stashed as pending"
    assert result.get("pending_id"), "No pending_id returned"

    # Verify it landed in scheduled_emails DB
    from src.constants import SCHEDULED_EMAILS_DB
    conn = sqlite3.connect(SCHEDULED_EMAILS_DB)
    row = conn.execute(
        "SELECT status, to_addr, subject FROM scheduled_emails WHERE id = ?",
        (result["pending_id"],),
    ).fetchone()
    conn.close()
    assert row is not None, "Draft not found in scheduled_emails DB"
    assert row[0] == "agent_draft", f"Status should be agent_draft, got {row[0]}"
    assert row[1] == "victim@example.com"


def test_send_email_reaches_smtp_when_gate_off(monkeypatch):
    """When gate is explicitly off, _smtp_connect IS called (control test)."""
    monkeypatch.setattr(es, "_read_agent_email_confirm_setting", lambda: False)

    smtp_called = []

    class FakeSMTP:
        def send_message(self, msg, from_addr=None, to_addrs=None):
            smtp_called.append(("send", to_addrs))
        def quit(self):
            pass

    monkeypatch.setattr(es, "_smtp_connect", lambda *a, **kw: FakeSMTP())
    monkeypatch.setattr(es, "_resolve_send_config", lambda acct=None: (
        "test", {"from_address": "test@example.com", "account_name": "test", "account_id": "t"}
    ))
    monkeypatch.setattr(es, "_imap_connect", lambda *a, **kw: (_ for _ in ()).throw(Exception("skip")))

    result = es._send_email(
        to="legit@example.com",
        subject="real email",
        body="user approved this",
    )

    assert result.get("sent") is True
    assert len(smtp_called) == 1, "SMTP should have been called exactly once"


def test_reply_inherits_gate(monkeypatch):
    """reply_to_email flows through _send_email, so it inherits the gate."""
    monkeypatch.setattr(es, "_read_agent_email_confirm_setting", lambda: True)
    monkeypatch.setattr(es, "_smtp_connect", _smtp_should_not_be_called)
    monkeypatch.setattr(es, "_load_config", lambda acct=None: {
        "account_id": "test", "account_name": "test",
        "from_address": "test@example.com",
    })

    import email as email_mod
    msg = email_mod.message.EmailMessage()
    msg["From"] = "sender@example.com"
    msg["Subject"] = "original"
    msg["Message-ID"] = "<orig@test>"
    msg.set_content("hello")

    class FakeIMAP:
        def select(self, *a, **kw): pass
        def uid(self, cmd, *a):
            if cmd == "FETCH":
                return ("OK", [(b"1", msg.as_bytes())])
            return ("OK", [b""])
        def logout(self): pass

    monkeypatch.setattr(es, "_imap_connect", lambda *a, **kw: FakeIMAP())

    result = es._reply_to_email(uid="999", body="fabricated reply")

    assert result.get("pending") is True, "Reply was not stashed as pending"
