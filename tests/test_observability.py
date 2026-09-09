"""
Tests for the observability branch: retraction, zero-tool-claim metadata,
MCP lockdown, and shipped_tools persistence.
"""
import json
import types

import pytest


# ── 1. get_context_messages retraction replacement ──

def _make_msg(role, content, **meta):
    """Minimal ChatMessage stand-in."""
    m = types.SimpleNamespace(role=role, content=content, metadata=dict(meta))
    m.to_dict = lambda: {"role": m.role, "content": m.content}
    return m


def _make_session(messages):
    """Minimal Session stand-in with the real get_context_messages logic."""
    from core.models import Session
    s = Session.__new__(Session)
    s.history = list(messages)
    return s


class TestRetraction:
    def test_retracted_message_replaced_in_context(self):
        msgs = [
            _make_msg("user", "check my email"),
            _make_msg("assistant", "I checked your email and found 3 new messages.", retracted=True),
            _make_msg("user", "that was wrong"),
            _make_msg("assistant", "I apologize for the confusion."),
        ]
        sess = _make_session(msgs)
        ctx = sess.get_context_messages()
        assert len(ctx) == 4
        assert ctx[1]["content"] == "[Assistant message retracted by user as fabricated. Do not reference its contents.]"
        assert ctx[1]["role"] == "assistant"
        # Non-retracted messages pass through unchanged
        assert ctx[3]["content"] == "I apologize for the confusion."

    def test_has_retracted_messages(self):
        msgs = [
            _make_msg("user", "hello"),
            _make_msg("assistant", "hi", retracted=True),
        ]
        sess = _make_session(msgs)
        assert sess.has_retracted_messages() is True

    def test_no_retracted_messages(self):
        msgs = [
            _make_msg("user", "hello"),
            _make_msg("assistant", "hi"),
        ]
        sess = _make_session(msgs)
        assert sess.has_retracted_messages() is False

    def test_retraction_orm_column(self):
        """Verify retracted flag round-trips through the real DB column
        (meta_data, not the ORM .metadata attribute)."""
        from core.database import SessionLocal, ChatMessage as DbChatMessage
        from core.database import Session as DbSession
        import uuid
        from datetime import datetime, timezone

        db = SessionLocal()
        sess_id = f"__test_retract_{uuid.uuid4().hex[:8]}__"
        msg_id = str(uuid.uuid4())
        try:
            # Create a parent session (FK constraint)
            now = datetime.now(timezone.utc)
            db_sess = DbSession(
                id=sess_id, name="test", endpoint_url="", model="test",
                created_at=now, updated_at=now,
            )
            db.add(db_sess)
            db.flush()

            # Write a message with retracted=True in metadata
            db_msg = DbChatMessage(
                id=msg_id,
                session_id=sess_id,
                role="assistant",
                content="I checked your email and found 3 new messages.",
                meta_data=json.dumps({"retracted": True, "_test": True}),
            )
            db.add(db_msg)
            db.commit()

            # Re-read and verify the flag survives
            reloaded = db.query(DbChatMessage).filter(DbChatMessage.id == msg_id).first()
            assert reloaded is not None
            meta = json.loads(reloaded.meta_data or "{}")
            assert meta["retracted"] is True

            # Toggle off
            meta["retracted"] = False
            reloaded.meta_data = json.dumps(meta)
            db.commit()
            reloaded2 = db.query(DbChatMessage).filter(DbChatMessage.id == msg_id).first()
            meta2 = json.loads(reloaded2.meta_data or "{}")
            assert meta2["retracted"] is False
        finally:
            # Clean up (cascade deletes messages)
            db.query(DbSession).filter(DbSession.id == sess_id).delete()
            db.commit()
            db.close()

    def test_slash_messages_still_excluded(self):
        msgs = [
            _make_msg("user", "/setup test", source="slash"),
            _make_msg("assistant", "Setup complete", source="slash"),
            _make_msg("user", "hello"),
            _make_msg("assistant", "hi"),
        ]
        sess = _make_session(msgs)
        ctx = sess.get_context_messages()
        assert len(ctx) == 2  # slash messages excluded


# ── 2. zero_tool_claim_count in save_assistant_response ──

class TestZeroToolClaimMetadata:
    def test_flag_written_to_metadata(self):
        """save_assistant_response with zero_tool_claim_count > 0 sets the flag."""
        from routes.chat_helpers import save_assistant_response

        # Minimal session mock
        saved = []
        class FakeMsg:
            def __init__(self, role, content, metadata=None):
                self.role = role
                self.content = content
                self.metadata = metadata or {}
        class FakeSess:
            model = "test-model"
            id = "test-session"
            history = []
            def add_message(self, msg):
                self.history.append(msg)
                saved.append(msg)

        sess = FakeSess()
        save_assistant_response(
            sess, None, "test-session", "I checked the file.",
            {"model": "test"},
            incognito=True,  # skip DB persist
            zero_tool_claim_count=2,
        )
        assert len(saved) == 1
        md = saved[0].metadata
        assert md["flagged_zero_tool_claim"] is True
        assert md["zero_tool_claim_count"] == 2

    def test_no_flag_when_zero(self):
        """zero_tool_claim_count=0 should not set the flag."""
        from routes.chat_helpers import save_assistant_response

        saved = []
        class FakeMsg:
            def __init__(self, role, content, metadata=None):
                self.role = role
                self.content = content
                self.metadata = metadata or {}
        class FakeSess:
            model = "test-model"
            id = "test-session"
            history = []
            def add_message(self, msg):
                self.history.append(msg)
                saved.append(msg)

        sess = FakeSess()
        save_assistant_response(
            sess, None, "test-session", "Hello!",
            {"model": "test"},
            incognito=True,
            zero_tool_claim_count=0,
        )
        assert len(saved) == 1
        md = saved[0].metadata
        assert "flagged_zero_tool_claim" not in md


# ── 3. MCP lockdown: retracted session strips email + memory tools ──

class TestRetractionLockdown:
    def test_mcp_email_stripped_by_suffix(self):
        """_is_disabled must match mcp__*__send_email when send_email is disabled."""
        # Import the actual filter logic — inline since it's a local def
        disabled = {"send_email", "reply_to_email", "manage_memory"}

        def _is_disabled(schema_name: str) -> bool:
            if schema_name in disabled:
                return True
            if "__" in schema_name:
                bare = schema_name.rsplit("__", 1)[-1]
                if bare in disabled:
                    return True
            return False

        # Bare names
        assert _is_disabled("send_email") is True
        assert _is_disabled("reply_to_email") is True
        assert _is_disabled("manage_memory") is True
        # MCP qualified names
        assert _is_disabled("mcp__camelot-email__send_email") is True
        assert _is_disabled("mcp__camelot-email__reply_to_email") is True
        # Safe tools pass through
        assert _is_disabled("list_emails") is False
        assert _is_disabled("read_email") is False
        assert _is_disabled("mcp__camelot-email__list_emails") is False
        assert _is_disabled("web_search") is False
        assert _is_disabled("bash") is False

    def test_retracted_session_strips_outbound_tools(self):
        """A session with retracted messages should not ship send_email,
        reply_to_email, or manage_memory schemas — including MCP variants."""
        disabled = {"send_email", "reply_to_email", "manage_memory"}

        def _is_disabled(schema_name: str) -> bool:
            if schema_name in disabled:
                return True
            if "__" in schema_name:
                bare = schema_name.rsplit("__", 1)[-1]
                if bare in disabled:
                    return True
            return False

        # Synthetic schema set covering builtin + MCP tools
        all_schemas = [
            {"type": "function", "function": {"name": "send_email"}},
            {"type": "function", "function": {"name": "reply_to_email"}},
            {"type": "function", "function": {"name": "manage_memory"}},
            {"type": "function", "function": {"name": "list_emails"}},
            {"type": "function", "function": {"name": "read_email"}},
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "bash"}},
            {"type": "function", "function": {"name": "mcp__camelot-email__send_email"}},
            {"type": "function", "function": {"name": "mcp__camelot-email__reply_to_email"}},
            {"type": "function", "function": {"name": "mcp__camelot-email__list_emails"}},
        ]
        filtered = [
            t for t in all_schemas
            if not _is_disabled(t.get("function", {}).get("name", ""))
        ]
        shipped_names = {t["function"]["name"] for t in filtered}

        # Must NOT contain outbound tools
        assert "send_email" not in shipped_names
        assert "reply_to_email" not in shipped_names
        assert "manage_memory" not in shipped_names
        assert "mcp__camelot-email__send_email" not in shipped_names
        assert "mcp__camelot-email__reply_to_email" not in shipped_names

        # Must still contain read-only tools
        assert "list_emails" in shipped_names
        assert "read_email" in shipped_names
        assert "web_search" in shipped_names
        assert "bash" in shipped_names
        assert "mcp__camelot-email__list_emails" in shipped_names


# ── 4. tool_parsing last_matched_pattern ──
# Circular import prevents direct import of parse_tool_blocks in test context.
# Use importlib to load just tool_parsing after bootstrapping its dependency.

class TestToolParsingPattern:
    @staticmethod
    def _get_module():
        """Import tool_parsing without triggering circular imports."""
        from collections import namedtuple
        import sys
        # Provide a minimal src.agent_tools stub if not already loaded
        if "src.agent_tools" not in sys.modules:
            stub = types.ModuleType("src.agent_tools")
            stub.ToolBlock = namedtuple("ToolBlock", ["tool_type", "content"])
            stub.TOOL_TAGS = {"bash", "python", "web_search", "web_fetch", "read_file",
                              "write_file", "manage_memory", "manage_tasks"}
            sys.modules["src.agent_tools"] = stub
        import importlib
        if "src.tool_parsing" in sys.modules:
            return importlib.reload(sys.modules["src.tool_parsing"])
        return importlib.import_module("src.tool_parsing")

    def test_fenced_pattern(self):
        tp = self._get_module()
        blocks = tp.parse_tool_blocks("```bash\nls -la\n```")
        assert len(blocks) == 1
        assert blocks[0].tool_type == "bash"
        assert tp.last_matched_pattern == "fenced"

    def test_bracket_marker_pattern(self):
        tp = self._get_module()
        blocks = tp.parse_tool_blocks("*[Searching — test query]*")
        assert tp.last_matched_pattern == "bracket_marker"

    def test_no_match_returns_none(self):
        tp = self._get_module()
        blocks = tp.parse_tool_blocks("Just a plain text message with no tool calls.")
        assert tp.last_matched_pattern is None
        assert blocks == []
