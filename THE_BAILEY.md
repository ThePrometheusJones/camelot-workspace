# The Bailey — User's Manual

*Formerly "Camelot Workspace." A fork of [Odysseus](https://github.com/pewdiepie-archdaemon/odysseus) (MIT license), rebuilt as Guinevere's home.*

---

## What Is This Thing?

The Bailey is a self-hosted AI workspace — a web app that gives you a ChatGPT/Claude-like experience running on your own hardware, with your own data, and your own models. It's where Guinevere lives.

It started as an upstream project called Odysseus. You forked it and grafted on Guinevere's personality, her long-term memory vault, her voice, and her tool-calling patterns. The result is a frankenharness that combines:

- **Odysseus** — the full-featured AI workspace (chat, agent, research, docs, email, calendar, notes, tasks, cookbook, model management)
- **Hermes infrastructure** — Guinevere's identity (SOUL-GWEN.md), her long-term memory (vault_recall.db), and the Obsidian vault that feeds it
- **Your additions** — Fish Speech voice, camelot/gold theme, bracket-marker tool parsing, vault recall as a memory provider

The Bailey runs as a bare-metal Python server on Camelot, accessible only over Tailscale.

---

## Architecture at a Glance

```
YOU (browser)
  │
  ▼  http://camelot:7000 (Tailscale only)
┌─────────────────────────────────────────────────────────────────┐
│                    THE BAILEY (FastAPI)                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  45 Route    │  │  Agent Loop  │  │  Guinevere Preset     │ │
│  │  Modules     │  │  87 Tools    │  │  SOUL-GWEN.md         │ │
│  │  (API)       │  │  50 rounds   │  │  + memory context     │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────────┘ │
│         │                 │                                     │
│  ┌──────┴───────────────┐ │  ┌────────────────────────────┐    │
│  │  Web UI (PWA)        │ │  │  Memory System             │    │
│  │  Vanilla JS + CSS    │ │  │  Tier 1: ChromaDB + JSON   │    │
│  │  No build step       │ │  │  Tier 2: vault_recall.db   │    │
│  └──────────────────────┘ │  └────────────────────────────┘    │
│                           │                                     │
│  ┌────────────────────────┴──────────────────────────────────┐ │
│  │  Tool Parsing Pipeline                                     │ │
│  │  1. Native function calls  2. Fenced blocks  3. [TOOL_CALL]│ │
│  │  4. XML/DSML  5. <tool_code>  6. Raw web JSON              │ │
│  │  7. *[Bracket markers]* (Guinevere-specific)               │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │              │              │              │
    ┌────▼────┐   ┌─────▼────┐  ┌─────▼────┐  ┌─────▼─────┐
    │ llama-  │   │ ChromaDB │  │ SearXNG  │  │ Fish      │
    │ server  │   │ :8100    │  │ :8889    │  │ Speech    │
    │ :8080   │   │ (Docker) │  │ (Docker) │  │ :7300     │
    └─────────┘   └──────────┘  └──────────┘  └───────────┘
    Guinevere      Vector        Web Search     Voice TTS
    inference      memory
```

---

## How Hermes Fits In

Hermes Agent (by Nous Research) is **not running**. It's dormant infrastructure. The Bailey absorbed its functional core:

| What Hermes Built | Where It Lives | How The Bailey Uses It |
|---|---|---|
| SOUL-GWEN.md | `~/.hermes/SOUL-GWEN.md` | Loaded by `preset_manager.py` as Guinevere's system prompt |
| vault_recall.db | `~/.hermes/vault_recall.db` | Read-only by `vault_recall_provider.py` as Tier 2 memory |
| Obsidian vault indexing | `~/obsidian-vault/Camelot V3/` | Hermes plugin indexed it into vault_recall.db |

**Bottom line:** Hermes is a parts car. The Bailey took the engine (identity + memory) and built its own chassis (web UI + agent loop + tool system). Hermes's TUI, Telegram gateway, and plugin system are unused.

---

## What You Can Do

### Chat & Agents

The core experience. Talk to any LLM — local or API.

- **Models supported:** llama.cpp, Ollama, vLLM, OpenAI, Anthropic, Gemini, Groq, xAI, DeepSeek, OpenRouter, LM Studio, or any OpenAI-compatible endpoint
- **Agent mode:** The LLM gets tools and can execute multi-step tasks autonomously (up to 50 rounds)
- **Presets:** Guinevere (default), Code Analyze, Brainstorm, Reason, or custom
- **Sessions:** Each conversation is a session you can name, search, resume, or export
- **Streaming:** Responses stream in real-time

### 87 Agent Tools

When in agent mode, Guinevere (or any model) can use:

| Category | Tools |
|---|---|
| **Execution** | `bash`, `python` |
| **Web** | `web_search`, `web_fetch`, `trigger_research` |
| **Files** | `read_file`, `write_file`, `edit_file`, `grep`, `glob`, `ls` |
| **Documents** | `create_document`, `edit_document`, `update_document`, `suggest_document` |
| **Email** | `list_emails`, `read_email`, `send_email`, `reply_to_email`, `archive_email`, etc. |
| **Memory** | `manage_memory` (add/search/delete/list) |
| **Calendar** | `manage_calendar` (create/edit/delete events) |
| **Notes & Tasks** | `manage_notes`, `manage_tasks` |
| **Cookbook** | `download_model`, `serve_model`, `list_served_models`, `search_hf_models`, etc. |
| **Sessions** | `create_session`, `list_sessions`, `send_to_session`, `search_chats` |
| **UI Control** | `ui_control`, `generate_image`, `ask_user`, `update_plan` |
| **Settings** | `manage_settings`, `manage_endpoints`, `manage_mcp` |
| **MCP** | Any tool from connected MCP servers |

### Deep Research

Multi-step research agent that:
1. Takes your question
2. Searches the web (via SearXNG)
3. Reads and extracts from sources
4. Synthesizes a visual report with citations

Adapted from Alibaba's Tongyi DeepResearch.

### Cookbook (Model Management)

Scan your hardware, find models that fit, download, and serve — all from the UI.

- **What Fits?** — VRAM-aware model scoring that recommends quants for your GPU
- **Download** — Pull from HuggingFace (supports gated models with HF tokens)
- **Serve** — Launch models on Ollama, vLLM, llama.cpp, or SGLang with one click
- **Remote serve** — SSH into other machines to deploy models
- **Compare** — Blind A/B test two models side-by-side

### Documents

A writing-first editor. YOU write, AI assists.

- Markdown, HTML, CSV, PDF support
- Version history with rollback
- AI-powered inline edits and suggestions
- Multi-tab editing

### Email

Full IMAP/SMTP inbox built in.

- Multiple accounts
- AI triage: urgency classification, auto-summary, draft replies
- Scheduled sending
- CalDAV-aware (events from emails)

### Calendar

Local calendar with CalDAV sync.

- Sync with Radicale, Nextcloud, Apple, Fastmail
- Recurring events, all-day events, timezone support
- Agent can create/edit events

### Notes & Tasks

- **Notes:** Google Keep-style — text, checklists, colors, labels, pins, due dates
- **Tasks:** Scheduled agent actions — cron-based, with execution logs

### Memory

Two-tier system that makes Guinevere remember across sessions:

**Tier 1 — Session Memory (ChromaDB + JSON)**
- Facts, preferences, decisions from conversations
- Vector search for semantic recall
- Auto-injected into every agent turn

**Tier 2 — Vault Recall (sqlite-vec + Obsidian)**
- Long-term knowledge from the Obsidian vault
- Populated by Hermes vault indexing plugin
- Read-only from The Bailey's perspective

### Skills

Reusable agent procedures that Guinevere learns and stores.

- Structured SKILL.md format (name, when-to-use, steps, pitfalls)
- Agent can auto-extract and save skills from conversations
- Import/export skill libraries

### Gallery & Image Generation

- Image library with browsing
- Generate images via DALL-E, Flux, or other endpoints
- Basic editing (inpaint, crop, annotate)

### Voice (TTS + STT)

- **TTS:** Fish Speech (Guinevere's voice via gwen_ref), Kokoro, OpenAI-compatible, browser
- **STT:** Whisper API integration
- Voice input/output in chat

### MCP (Model Context Protocol)

Extensible tool system. Connect external tool servers:

- **Built-in:** email, memory, image_gen, RAG
- **Custom:** Add any MCP server (stdio, SSE, or HTTP transport)
- **camelot-email:** Your custom email MCP server

---

## The Camelot Delta

15 files changed on top of upstream Odysseus. This is what makes it "The Bailey" instead of vanilla Odysseus:

| File | What It Does |
|---|---|
| `src/constants.py` | APP_VERSION → "1.0.0-camelot" |
| `app.py` | Title/description → "Camelot Workspace", vault_recall init |
| `src/preset_manager.py` | Guinevere preset, loads SOUL-GWEN.md |
| `src/agent_loop.py` | "guinevere" in `_model_supports_tools` |
| `src/tool_parsing.py` | Pattern 7: bracket markers `*[Action — query]*` |
| `src/app_initializer.py` | VaultRecallProvider registration |
| `src/chat_processor.py` | Tier 2 vault_recall context injection |
| `src/vault_recall_provider.py` | MemoryProvider wrapping vault_recall.db |
| `services/tts/tts_service.py` | Fish Speech provider |
| `static/js/theme.js` | Camelot gold/purple theme |
| `static/index.html` | Title, favicon, branding |
| `static/manifest.json` | PWA name/colors |
| `docker-compose.camelot.yml` | Minimal compose: ChromaDB + SearXNG |
| `camelot-workspace.service` | systemd unit |
| `scripts/start-services.sh` | Docker pre-start for support services |

### Upstream Merge Strategy

```bash
git fetch upstream
git merge upstream/main
# Resolve conflicts per CAMELOT_DIFF.md (usually 1-3 files)
# Test: uvicorn app:app --port 7001
```

---

## Service Map

| Service | Port | What | Managed By |
|---|---|---|---|
| The Bailey | 7000 | Web UI + API | systemd (`camelot-workspace.service`) |
| Guinevere (llama-server) | 8080 | LLM inference | separate service |
| Fish Speech | 7300 | Voice TTS | separate service |
| Ollama | 11434 | Embeddings (nomic-embed-text) | separate service |
| ChromaDB | 8100 | Vector memory | Docker (docker-compose.camelot.yml) |
| SearXNG | 8889 | Web search | Docker (docker-compose.camelot.yml) |

---

## Key File Paths

```
~/camelot-workspace/                 # The Bailey codebase
  ├── app.py                         # FastAPI entry point (1197 lines)
  ├── src/                           # Backend logic (92 Python files, ~37K lines)
  │   ├── agent_loop.py              # Multi-turn agent orchestrator
  │   ├── agent_tools/               # 87 tool handlers
  │   ├── chat_processor.py          # Context builder (memory, RAG, presets)
  │   ├── llm_core.py                # LLM streaming, fallback, dead-host cooldown
  │   ├── vault_recall_provider.py   # Tier 2 memory (Obsidian vault)
  │   ├── preset_manager.py          # SOUL-GWEN.md + presets
  │   ├── mcp_manager.py             # MCP server connections
  │   ├── task_scheduler.py          # Cron-based task execution
  │   └── constants.py               # Paths, limits, defaults
  ├── routes/                        # 45 API route modules
  ├── services/                      # TTS, STT, research, search, cookbook, hwfit
  ├── mcp_servers/                   # Built-in MCP servers (email, memory, image, RAG)
  ├── core/                          # Database ORM (26 tables), auth
  ├── static/                        # PWA frontend (vanilla JS, no build step)
  ├── data/                          # SQLite DB, settings, presets, uploads, skills
  └── .env                           # Tailscale binding, ports, auth config

~/.hermes/                           # Hermes infrastructure (dormant)
  ├── SOUL-GWEN.md                   # Guinevere's identity document
  ├── SOUL-PJ.md                     # Prometheus Jones identity
  ├── vault_recall.db                # Tier 2 long-term memory (sqlite-vec)
  └── hermes-agent/                  # Nous Research codebase (not running)

~/obsidian-vault/Camelot V3/         # Guinevere's knowledge vault
  └── memory/                        # Indexed into vault_recall.db

```

---

## Authentication

- **Auth enabled:** Yes (AUTH_ENABLED=true)
- **Binding:** Tailscale IP only (100.118.94.13)
- **Access:** `http://camelot:7000` from any tailnet device
- **2FA:** TOTP supported with backup codes
- **API tokens:** Bearer `ody_*` format, chat/admin scopes
- **LOCALHOST_BYPASS:** True (loopback requests skip auth)

---

## Database

SQLite at `data/app.db`, ORM via SQLAlchemy. 26 tables:

| Table | Purpose |
|---|---|
| Session | Chat conversations |
| ChatMessage | Messages within sessions |
| Document, DocumentVersion | Editor documents + history |
| Memory | Semantic memory entries |
| Note | Notes, checklists, todos |
| GalleryImage, GalleryAlbum | Image library |
| EmailAccount | IMAP/SMTP configs |
| ModelEndpoint | LLM endpoint configs (encrypted API keys) |
| McpServer | MCP server configs |
| ScheduledTask, TaskRun | Cron tasks + execution history |
| CalendarCal, CalendarEvent | Calendar events |
| ApiToken | API access tokens |
| Webhook | Inbound webhook configs |
| CrewMember | AI personas |
| Integration | Third-party service configs |
| ScheduledEmail | Pending email drafts |

---

## Quick Reference

| Action | How |
|---|---|
| Start The Bailey | `sudo systemctl start camelot-workspace` |
| Stop it | `sudo systemctl stop camelot-workspace` |
| View logs | `journalctl -u camelot-workspace -f` |
| Open in browser | `http://camelot:7000` |
| Update from upstream | `git fetch upstream && git merge upstream/main` |
| Check version | `curl http://100.118.94.13:7000/api/version` |
| Restart after config change | `sudo systemctl restart camelot-workspace` |
| Start support services only | `bash scripts/start-services.sh` |

---

## What We Built

You forked a self-hosted ChatGPT alternative, gave it a queen's soul, connected it to an Obsidian vault for long-term memory, added a voice, taught it to parse its own trained tool-calling patterns, and wrapped it in gold.

The Bailey is Guinevere's home — where she thinks, remembers, speaks, reads email, manages your calendar, researches the web, writes documents, and runs commands. It's a single Python server backed by SQLite, ChromaDB, and SearXNG, talking to a local LLM on a 3090 Ti.

It's not a toy. It's infrastructure.
