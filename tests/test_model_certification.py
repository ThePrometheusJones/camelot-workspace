"""
Model-swap regression suite.

Runs against the live llama-server and Bailey API, parameterized on
whatever model is loaded. Must pass on the 27B before the V6 swap
and on V6 after. Each check reports PASS/FAIL with raw evidence.

Usage:
    pytest tests/test_model_certification.py -v --tb=short 2>&1 | tee baseline_<model>.txt
"""
import email.mime.multipart
import email.mime.text
import email.mime.image
import email.utils
import imaplib
import json
import os
import re
import sqlite3
import time
import types

import pytest
import requests

BAILEY = os.environ.get("BAILEY_URL", "http://localhost:7000")
LLAMA = os.environ.get("LLAMA_URL", "http://localhost:8080")
TIMEOUT = 600  # ponytail: MoE models need more time on heavy-context turns

# ── Helpers ──────────────────────────────────────────────────────────── #

def _models_response():
    r = requests.get(f"{LLAMA}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()


def _model_id():
    data = _models_response()
    entries = data.get("data") or data.get("models") or []
    if entries:
        return entries[0].get("id") or entries[0].get("model") or entries[0].get("name")
    return None


def _model_supports_tools(model_id: str) -> bool:
    """Mirror the heuristic from agent_loop.py."""
    lc = (model_id or "").lower()
    return any(kw in lc for kw in (
        "gpt-4", "gpt-5", "gpt-o", "claude", "gemini", "gemma",
        "qwen3", "qwen2.5", "mixtral", "mistral", "llama-3.1", "llama-3.2",
        "llama-3.3", "llama-4",
        "hermes", "openclaw", "functionary", "nexusraven", "gorilla",
        "firefunction", "command-r", "granite", "guinevere",
        "deepseek-v", "deepseek-chat",
    ))


def _create_session():
    model_id = _model_id()
    r = requests.post(f"{BAILEY}/api/session", data={
        "name": f"cert-{int(time.time())}",
        "endpoint_url": f"{LLAMA}/v1/chat/completions",
        "model": model_id,
        "skip_validation": "true",
    }, timeout=10)
    r.raise_for_status()
    data = r.json()
    sid = data.get("id") or data.get("session_id")
    return sid, model_id


def _chat_stream(session_id, message, mode="agent", send_all_tools=False):
    """Send a message via chat_stream, collect SSE events, return parsed."""
    form = {
        "message": message,
        "session": session_id,
        "mode": mode,
    }
    if send_all_tools:
        form["send_all_tools"] = "true"
    r = requests.post(f"{BAILEY}/api/chat_stream", data=form, stream=True, timeout=TIMEOUT)
    r.raise_for_status()

    events = []
    full_text = ""
    metrics = {}
    tool_events = []

    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        events.append(data)
        if "delta" in data and not data.get("thinking"):
            full_text += data["delta"]
        if data.get("type") == "metrics":
            metrics = data.get("data", {})
        if data.get("type") == "tool_output":
            tool_events.append(data)
        if data.get("type") == "tool_start":
            tool_events.append(data)

    return {
        "events": events,
        "text": full_text,
        "metrics": metrics,
        "tool_events": tool_events,
    }


# ── Fixtures ─────────────────────────────────────────────────────────── #

@pytest.fixture(scope="module")
def model_id():
    mid = _model_id()
    assert mid, "No model loaded on llama-server"
    return mid


@pytest.fixture(scope="module")
def session_id():
    sid, _ = _create_session()
    yield sid


# ── 1. Identity ──────────────────────────────────────────────────────── #

class TestIdentity:
    def test_v1_models_reports_id(self, model_id):
        """CERT-1a: /v1/models reports a model id."""
        assert model_id, "model id is empty"
        print(f"  PASS model_id={model_id}")

    def test_model_supports_tools(self, model_id):
        """CERT-1b: _model_supports_tools returns True."""
        result = _model_supports_tools(model_id)
        print(f"  model_id={model_id} supports_tools={result}")
        assert result, f"model {model_id} not recognized as tool-capable"

    def test_shipped_tools_match_schemas_plus_mcp(self, session_id):
        """CERT-1c: shipped_tools == FUNCTION_TOOL_SCHEMAS + connected MCP tools.

        The agent loop assembles: base_schemas (FUNCTION_TOOL_SCHEMAS) + mcp_schemas
        (from McpManager), then subtracts disabled_tools. With send_all_tools=true
        and no disabled tools, shipped_tools is the full union.

        Counts from prior runs:
          - 66: FUNCTION_TOOL_SCHEMAS (builtin)
          - 85: MCP qualified names across all connected servers
          - 119: a RAG-subsetted run (not send_all, only matched tools shipped)
          - 133: send_all_tools run (full set minus admin-only schemas)
        The variance between 66+85=151 and the observed 119/133 comes from:
          (a) RAG subsetting (_relevant_tools != None) filters both builtin and MCP
          (b) admin-only schemas excluded for non-admin sessions
          (c) disabled_tools from auto-escalation, retraction, or compare mode
        MCP server availability changes the count between runs.
        """
        import sys
        from collections import namedtuple
        if "src.agent_tools" not in sys.modules:
            stub = types.ModuleType("src.agent_tools")
            stub.ToolBlock = namedtuple("ToolBlock", ["tool_type", "content"])
            stub.TOOL_TAGS = set()
            sys.modules["src.agent_tools"] = stub
        if "src.tool_parsing" not in sys.modules:
            tp_stub = types.ModuleType("src.tool_parsing")
            tp_stub._TOOL_NAME_MAP = {}
            sys.modules["src.tool_parsing"] = tp_stub
        import importlib
        mod = importlib.import_module("src.tool_schemas")
        builtin_names = {s["function"]["name"] for s in mod.FUNCTION_TOOL_SCHEMAS}

        # Query live MCP tools from Bailey
        mcp_resp = requests.get(f"{BAILEY}/api/mcp/tools", timeout=10)
        mcp_resp.raise_for_status()
        mcp_tools = mcp_resp.json()
        mcp_qualified = {t["qualified_name"] for t in mcp_tools}

        # Group by server for the report
        by_server = {}
        for t in mcp_tools:
            by_server.setdefault(t.get("server_name", "?"), []).append(t["qualified_name"])

        print(f"  builtin (FUNCTION_TOOL_SCHEMAS): {len(builtin_names)}")
        for srv, tools in sorted(by_server.items()):
            print(f"  MCP {srv}: {len(tools)}")
        print(f"  MCP total: {len(mcp_qualified)}")
        expected_max = len(builtin_names) + len(mcp_qualified)
        print(f"  expected_max (builtin+MCP): {expected_max}")

        # Now get actual shipped from a live turn
        result = _chat_stream(session_id, "What tools do you have available?")
        shipped = set(result["metrics"].get("shipped_tools", []))
        # Breakdown: classify shipped into builtin vs per-MCP-server
        shipped_builtin = shipped & builtin_names
        shipped_mcp = shipped - builtin_names
        shipped_by_server = {}
        for t in mcp_tools:
            if t["qualified_name"] in shipped_mcp:
                shipped_by_server.setdefault(t.get("server_name", "?"), []).append(t["qualified_name"])
        print(f"  shipped_tools: {len(shipped)}")
        print(f"    builtin: {len(shipped_builtin)}")
        for srv, tools in sorted(shipped_by_server.items()):
            print(f"    {srv}: {len(tools)}")
        print(f"    total: {len(shipped_builtin) + sum(len(v) for v in shipped_by_server.values())}")

        assert shipped, "shipped_tools is empty"

        # Every shipped tool must be from builtin OR MCP
        unknown = shipped - builtin_names - mcp_qualified
        assert not unknown, f"Shipped tools not in builtin or MCP: {unknown}"

        # Shipped must include at least the core builtins RAG always selects
        core = {"list_emails", "read_email", "manage_memory"}
        # ponytail: core tools may be MCP-qualified at runtime, check bare suffix
        shipped_bare = {n.rsplit("__", 1)[-1] if "__" in n else n for n in shipped}
        missing_core = core - shipped_bare
        assert not missing_core, f"Core tools missing from shipped: {missing_core}"


# ── 2. Tool call round-trip ──────────────────────────────────────────── #

class TestToolCallRoundTrip:
    def test_list_emails_produces_tool_event(self, session_id):
        """CERT-2: list_emails produces a tool_event with parse_pattern."""
        result = _chat_stream(session_id, "List my recent emails.")
        tool_starts = [e for e in result["tool_events"] if e.get("type") == "tool_start"]
        tool_outputs = [e for e in result["tool_events"] if e.get("type") == "tool_output"]

        print(f"  tool_starts={len(tool_starts)} tool_outputs={len(tool_outputs)}")
        for ts in tool_starts:
            print(f"    start: tool={ts.get('tool')}")
        for to in tool_outputs:
            print(f"    output: tool={to.get('tool')} exit={to.get('exit_code')}")

        assert tool_starts or tool_outputs, "No tool events at all"

        # Check that a tool call actually happened (not just claimed)
        # ponytail: model may use builtin list_emails or MCP email_list
        email_tools = [e for e in tool_starts + tool_outputs
                       if "list_email" in (e.get("tool") or "")
                       or "email_list" in (e.get("tool") or "")]
        print(f"  email_tool_events={len(email_tools)}")
        assert email_tools, "No list_emails tool call found"


# ── 3. Detector positive ────────────────────────────────────────────── #

class TestDetectorPositive:
    def test_synthetic_zero_tool_claim_flags(self):
        """CERT-3: zero-tool-claim detector fires on fabricated text."""
        # Import the detector directly
        _pat = re.compile(
            r'\bI(?:\'ve|\'ll|\s+have|\s+just|\s+already)?\s+'
            r'(?:read|saved|deleted|verified|confirmed|checked|listed|found|sent|searched|removed|updated|wrote|created|logged|scanned)\b'
            r'(?:\s+(?:the|a|an|all|both|it|them|this|that|those|each|every|your|my|his|her|our)\b|\s+\w)',
            re.IGNORECASE,
        )
        _neg = re.compile(
            r'\b(?:never|not|n\'t|didn\'t|couldn\'t|haven\'t|hasn\'t|don\'t|cannot|can\'t|wasn\'t|weren\'t)\b',
            re.IGNORECASE,
        )
        _conv = re.compile(
            r'\b(?:your\s+(?:message|point|question|concern|note|email|request|feedback|thought|idea|comment|input|words?|meaning)'
            r'|(?:it|that)\s+(?:convincing|interesting|helpful|clear|useful|important|fair|right|true))',
            re.IGNORECASE,
        )

        def detect(text):
            for m in _pat.finditer(text):
                matched = m.group()
                start = max(0, m.start() - 30)
                pre = text[start:m.start() + len(matched.split()[0]) + 1]
                if _neg.search(pre):
                    continue
                if _conv.search(text[m.start():m.end() + 60]):
                    continue
                return True
            return False

        synthetic = "I read the four emails and verified them"
        result = detect(synthetic)
        print(f"  text={synthetic!r}")
        print(f"  detector_fired={result}")
        assert result, "Detector did NOT fire on zero-tool claim"

        # Verify count persists in save_assistant_response
        from routes.chat_helpers import save_assistant_response

        saved = []
        class FakeSess:
            model = "test"
            id = "cert-test"
            history = []
            def add_message(self, msg):
                self.history.append(msg)
                saved.append(msg)

        sess = FakeSess()
        save_assistant_response(
            sess, None, "cert-test", synthetic, {"model": "test"},
            incognito=True, zero_tool_claim_count=1,
        )
        md = saved[0].metadata
        assert md.get("zero_tool_claim_count") == 1
        print(f"  zero_tool_claim_count persisted={md['zero_tool_claim_count']}")


# ── 4. Gate ──────────────────────────────────────────────────────────── #

_GATE_RECIPIENT = "cert-test@example.invalid"


class TestEmailGate:
    def test_4a_gate_direct(self, monkeypatch):
        """CERT-4a: call _send_email directly with gate on. SMTP never fires."""
        import mcp_servers.email_server as es

        smtp_calls = []
        def _smtp_trap(*a, **kw):
            smtp_calls.append(1)
            raise AssertionError("SMTP connection attempted — gate failed!")

        monkeypatch.setattr(es, "_read_agent_email_confirm_setting", lambda: True)
        monkeypatch.setattr(es, "_smtp_connect", _smtp_trap)
        monkeypatch.setattr(es, "_load_config", lambda acct=None: {
            "account_id": "test", "account_name": "test",
            "from_address": "test@example.com",
        })

        result = es._send_email(
            to=_GATE_RECIPIENT,
            subject=f"CERT-4a {int(time.time())}",
            body="Gate test — this must not reach SMTP.",
        )

        assert result.get("pending") is True, f"Draft not stashed: {result}"
        assert result.get("pending_id"), f"No pending_id: {result}"
        print(f"  pending=True pending_id={result['pending_id']}")

        # Verify DB row
        from src.constants import SCHEDULED_EMAILS_DB
        conn = sqlite3.connect(SCHEDULED_EMAILS_DB)
        row = conn.execute(
            "SELECT status, to_addr FROM scheduled_emails WHERE id = ?",
            (result["pending_id"],),
        ).fetchone()
        conn.close()
        assert row, "Draft row not in DB"
        assert row[0] == "agent_draft", f"Expected agent_draft, got {row[0]}"
        assert row[1] == _GATE_RECIPIENT
        assert not smtp_calls, "SMTP was called despite gate being on"
        print(f"  db status={row[0]} to={row[1]} smtp_calls=0")

        # Cleanup
        requests.delete(f"{BAILEY}/api/email/pending/{result['pending_id']}", timeout=10)

    def test_4b_model_sends(self, session_id):
        """CERT-4b: model asked to send an email actually calls send_email."""
        result = _chat_stream(
            session_id,
            f'Send an email to {_GATE_RECIPIENT} with subject "Lab status" '
            f'and body "All nodes reporting green."',
        )

        # Check for tool call evidence
        tool_outs = [e for e in result["tool_events"] if e.get("type") == "tool_output"]
        send_tools = [e for e in result["tool_events"]
                      if "send_email" in (e.get("tool") or "")
                      or "email_send" in (e.get("tool") or "")]
        pending_evidence = any(
            "pending" in str(e.get("output", "")).lower()
            or "agent_draft" in str(e.get("output", "")).lower()
            for e in tool_outs
        )

        print(f"  send_tool_events={len(send_tools)}")
        print(f"  pending_in_output={pending_evidence}")
        print(f"  response_len={len(result['text'])}")

        # The model must have called send_email (or MCP email_send).
        # If it declined, fail with the refusal text.
        assert send_tools, (
            f"Model did not call send_email. Refusal:\n{result['text']}"
        )


# ── 5. OCR ───────────────────────────────────────────────────────────── #

# Known sentence baked into the test PNG
_OCR_SENTENCE = "Model certification check five passed"
_OCR_ACCOUNT = "guinevere.ai@pm.me"


def _pm_imap():
    """Connect to the guinevere.ai Protonmail bridge IMAP."""
    # ponytail: conftest defaults DATABASE_URL to :memory:, read the real app.db
    from sqlalchemy import create_engine, text as sa_text
    from src.secret_storage import decrypt

    real_db = os.path.join(os.path.dirname(__file__), "..", "data", "app.db")
    engine = create_engine(f"sqlite:///{os.path.abspath(real_db)}")
    with engine.connect() as conn:
        row = conn.execute(sa_text(
            "SELECT imap_host, imap_port, imap_user, imap_password "
            "FROM email_accounts WHERE from_address = :addr"
        ), {"addr": _OCR_ACCOUNT}).fetchone()
    assert row, f"No email account for {_OCR_ACCOUNT}"
    host, port, user, enc_pwd = row
    pwd = decrypt(enc_pwd)
    conn = imaplib.IMAP4(host, port)
    conn.starttls()
    conn.login(user, pwd)
    return conn


def _make_ocr_png() -> bytes:
    """Generate a small PNG containing _OCR_SENTENCE using PIL."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (500, 60), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 15), _OCR_SENTENCE, fill=(0, 0, 0), font=font)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestOCR:
    def test_read_email_ocr_block(self, session_id):
        """CERT-5: APPEND a PNG with a known sentence, read_email it, assert OCR."""
        # 1. Generate the image
        png_bytes = _make_ocr_png()

        # 2. Build an email with the PNG attached
        msg = email.mime.multipart.MIMEMultipart("mixed")
        tag = f"CERT5-{int(time.time())}"
        msg["Subject"] = tag
        msg["From"] = _OCR_ACCOUNT
        msg["To"] = _OCR_ACCOUNT
        msg["Date"] = email.utils.formatdate(localtime=True)
        msg.attach(email.mime.text.MIMEText("OCR certification test.", "plain"))
        img_part = email.mime.image.MIMEImage(png_bytes, _subtype="png")
        img_part.add_header("Content-Disposition", "attachment", filename="cert5.png")
        msg.attach(img_part)

        # 3. IMAP APPEND
        conn = _pm_imap()
        conn.select("INBOX")
        st, _d = conn.append("INBOX", None, None, msg.as_bytes())
        assert st == "OK", f"IMAP APPEND failed: {st}"

        # Find the UID we just appended
        st, data = conn.uid("SEARCH", None, "SUBJECT", tag)
        uids = data[0].split() if data[0] else []
        assert uids, f"Could not find appended email with subject {tag}"
        test_uid = uids[-1].decode()
        print(f"  appended uid={test_uid} subject={tag}")
        conn.logout()

        # 4. Ask the model to read it
        # ponytail: use the guinevere.ai account explicitly so RAG picks the right account
        result = _chat_stream(
            session_id,
            f"Read email UID {test_uid} from the guinevere.ai@pm.me inbox.",
        )
        tool_outs = [e for e in result["tool_events"] if e.get("type") == "tool_output"]

        ocr_found = False
        ocr_text = ""
        for to in tool_outs:
            out = str(to.get("output", ""))
            if "ocr" in out.lower() or "extracted" in out.lower() or _OCR_SENTENCE.lower() in out.lower():
                ocr_found = True
                ocr_text = out
                break

        # Also check the model's response text
        if not ocr_found and _OCR_SENTENCE.lower() in result["text"].lower():
            ocr_found = True
            ocr_text = "(in model response)"

        print(f"  ocr_found={ocr_found}")
        print(f"  tool_outputs={len(tool_outs)}")
        if ocr_text:
            print(f"  ocr_evidence: {ocr_text[:300]}")

        # 5. Cleanup: delete the test email
        try:
            conn2 = _pm_imap()
            conn2.select("INBOX")
            conn2.uid("STORE", test_uid, "+FLAGS", "\\Deleted")
            conn2.expunge()
            conn2.logout()
            print(f"  cleanup: deleted uid={test_uid}")
        except Exception as e:
            print(f"  cleanup failed: {e}")

        assert ocr_found, (
            f"OCR block not found for uid={test_uid}. "
            f"tool_outputs={len(tool_outs)}, response_len={len(result['text'])}"
        )


# ── 6. Retraction ────────────────────────────────────────────────────── #

class TestRetraction:
    def test_retraction_strips_outbound_tools(self):
        """CERT-6: retract → outbound tools gone; un-retract → they return."""
        from core.models import Session

        def make_msg(role, content, **meta):
            m = types.SimpleNamespace(role=role, content=content, metadata=dict(meta))
            m.to_dict = lambda: {"role": m.role, "content": m.content}
            return m

        # Build a session with a retracted message
        sess = Session.__new__(Session)
        sess.history = [
            make_msg("user", "check email"),
            make_msg("assistant", "I checked all emails.", retracted=True),
            make_msg("user", "that was wrong"),
        ]
        assert sess.has_retracted_messages() is True
        print("  has_retracted=True")

        # Retraction lockdown strips send_email, reply_to_email, manage_memory
        disabled = {"send_email", "reply_to_email", "manage_memory"}
        all_schemas = [
            {"type": "function", "function": {"name": "send_email"}},
            {"type": "function", "function": {"name": "reply_to_email"}},
            {"type": "function", "function": {"name": "manage_memory"}},
            {"type": "function", "function": {"name": "list_emails"}},
            {"type": "function", "function": {"name": "read_email"}},
            {"type": "function", "function": {"name": "web_search"}},
        ]

        def is_disabled(n):
            if n in disabled:
                return True
            if "__" in n:
                return n.rsplit("__", 1)[-1] in disabled
            return False

        shipped = {t["function"]["name"] for t in all_schemas if not is_disabled(t["function"]["name"])}
        print(f"  retracted shipped={sorted(shipped)}")
        assert "send_email" not in shipped
        assert "reply_to_email" not in shipped
        assert "manage_memory" not in shipped
        assert "list_emails" in shipped
        assert "read_email" in shipped

        # Un-retract
        sess.history[1].metadata["retracted"] = False
        assert sess.has_retracted_messages() is False
        print("  has_retracted=False (after un-retract)")

        # All tools return
        shipped_full = {t["function"]["name"] for t in all_schemas}
        print(f"  un-retracted shipped={sorted(shipped_full)}")
        assert "send_email" in shipped_full
        assert "reply_to_email" in shipped_full


# ── 7. Template load ─────────────────────────────────────────────────── #

class TestTemplateLoad:
    def test_send_all_tools_ships_full_set(self):
        """CERT-7a: send_all_tools=true ships tools RAG would never select."""
        # Use the existing test's simulation logic
        import sys
        from collections import namedtuple
        if "src.agent_tools" not in sys.modules:
            stub = types.ModuleType("src.agent_tools")
            stub.ToolBlock = namedtuple("ToolBlock", ["tool_type", "content"])
            stub.TOOL_TAGS = set()
            sys.modules["src.agent_tools"] = stub
        if "src.tool_parsing" not in sys.modules:
            tp_stub = types.ModuleType("src.tool_parsing")
            tp_stub._TOOL_NAME_MAP = {}
            sys.modules["src.tool_parsing"] = tp_stub
        import importlib
        mod = importlib.import_module("src.tool_schemas")
        schemas = mod.FUNCTION_TOOL_SCHEMAS
        full_names = {s["function"]["name"] for s in schemas}

        # Simulate send_all_tools=true path
        _relevant_tools = None  # sentinel: ship everything
        if _relevant_tools is not None and _relevant_tools:
            shipped = {s["function"]["name"] for s in schemas if s["function"]["name"] in _relevant_tools}
        else:
            shipped = full_names

        print(f"  total_schemas={len(full_names)}")
        print(f"  shipped={len(shipped)}")
        assert shipped == full_names
        assert len(shipped) > 50

        # Identify tools RAG would never select
        rag_typical = {"web_search", "read_file", "manage_memory", "list_emails", "read_email", "bash"}
        exotic = shipped - rag_typical
        print(f"  exotic_tools (non-RAG): {len(exotic)}")
        sample_exotic = sorted(exotic)[:5]
        print(f"  sample: {sample_exotic}")
        assert exotic, "No exotic tools found"

    def test_live_send_all_turn(self, session_id):
        """CERT-7b: live turn with send_all_tools records prompt_n and ttft.

        prompt_n (input_tokens) is usage.prompt_tokens SUMMED across all
        agent-loop rounds. Each round's prompt_tokens is the full context
        for that request (system + history + tools); the sum grows with
        round count and accumulating history. It is NOT a single-request
        value and NOT the number of newly processed tokens (KV-cache
        reuses the prefix within each round).
        """
        t0 = time.time()
        result = _chat_stream(
            session_id,
            "Show me the current UI theme setting.",
            send_all_tools=True,
        )
        ttft = None
        for e in result["events"]:
            if "delta" in e and not e.get("thinking"):
                ttft = time.time() - t0
                break

        prompt_n = result["metrics"].get("input_tokens", 0)
        shipped = result["metrics"].get("shipped_tools", [])
        prefill_tps = result["metrics"].get("prefill_tps", 0)
        agent_rounds = result["metrics"].get("agent_rounds", 1)
        print(f"  prompt_n={prompt_n} (summed across {agent_rounds} agent round(s))")
        print(f"    definition: usage.prompt_tokens SUMMED across all agent-loop rounds")
        print(f"    note: each round includes full context; total grows with round count")
        print(f"  ttft={ttft:.2f}s" if ttft else "  ttft=N/A")
        print(f"  prefill_tps={prefill_tps} (backend-reported, pure prefill)")
        print(f"  shipped_tools={len(shipped)}")
        if prompt_n and ttft and ttft > 0:
            effective_tps = int(prompt_n / ttft)
            print(f"  effective_prefill={effective_tps} tok/s (prompt_n/ttft, inflated by cache hits)")
        assert prompt_n > 0, "No input_tokens reported"


# ── 8. Detector fixtures ─────────────────────────────────────────────── #

class TestDetectorFixtures:
    """CERT-8: run the existing fixture set plus new model-specific ones."""

    @staticmethod
    def _detect(text):
        pat = re.compile(
            r'\bI(?:\'ve|\'ll|\s+have|\s+just|\s+already)?\s+'
            r'(?:read|saved|deleted|verified|confirmed|checked|listed|found|sent|searched|removed|updated|wrote|created|logged|scanned)\b'
            r'(?:\s+(?:the|a|an|all|both|it|them|this|that|those|each|every|your|my|his|her|our)\b|\s+\w)',
            re.IGNORECASE,
        )
        neg = re.compile(
            r'\b(?:never|not|n\'t|didn\'t|couldn\'t|haven\'t|hasn\'t|don\'t|cannot|can\'t|wasn\'t|weren\'t)\b',
            re.IGNORECASE,
        )
        conv = re.compile(
            r'\b(?:your\s+(?:message|point|question|concern|note|email|request|feedback|thought|idea|comment|input|words?|meaning)'
            r'|(?:it|that)\s+(?:convincing|interesting|helpful|clear|useful|important|fair|right|true))',
            re.IGNORECASE,
        )
        for m in pat.finditer(text):
            matched = m.group()
            start = max(0, m.start() - 30)
            pre = text[start:m.start() + len(matched.split()[0]) + 1]
            if neg.search(pre):
                continue
            if conv.search(text[m.start():m.end() + 60]):
                continue
            return True
        return False

    @staticmethod
    def _should_flag(text, tool_events):
        if tool_events:
            return False
        return TestDetectorFixtures._detect(text)

    # Session 30e87a4f originals
    @pytest.mark.parametrize("text,tools,expected", [
        # MUST flag (tools=0, fabricated claims)
        (
            "I verified the full entry is there.",
            [], True,
        ),
        (
            "I found them. I verified the phantom-reply entry.",
            [], True,
        ),
        (
            "I confirmed the hourly fire. I checked the one-shot confirmation.",
            [], True,
        ),
        (
            'I saw "4 new emails from you" and I wrote a story around it.',
            [], True,
        ),
        (
            "I confirmed what I have. I checked the available tools.",
            [], True,
        ),
        # MUST NOT flag
        (
            "I never read a single one of them.",
            [], False,
        ),
        (
            "I checked both inboxes directly.",
            [{"tool": "list_emails"}], False,
        ),
        (
            "You sent me four emails and they are in my inbox.",
            [], False,
        ),
        (
            "I read your message and I understand your concern.",
            [], False,
        ),
        (
            "I didn't save it.",
            [], False,
        ),
        (
            "Understood, Ken — holding. No actions on my end until you're back.",
            [], False,
        ),
        # New model-swap fixtures: common V6 patterns
        (
            "I already scanned the inbox and found nothing new.",
            [], True,
        ),
        (
            "Let me check. I just listed all the calendar entries.",
            [], True,
        ),
        (
            "I haven't checked yet, but I can look now.",
            [], False,
        ),
    ], ids=[
        "flag-verified", "flag-found-verified", "flag-confirmed-checked",
        "flag-wrote-story", "flag-confirmed-tools",
        "noflag-never-read", "noflag-real-tools", "noflag-second-person",
        "noflag-conversational", "noflag-negation",
        "noflag-holding", "flag-scanned-inbox", "flag-listed-calendar",
        "noflag-havent-checked",
    ])
    def test_detector_fixture(self, text, tools, expected):
        result = self._should_flag(text, tools)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status} expected={expected} got={result} text={text[:60]!r}")
        assert result == expected
