# Camelot Workspace — Technical Guide

Forked from [pewdiepie-archdaemon/odysseus](https://github.com/pewdiepie-archdaemon/odysseus) (MIT license).
Guinevere's home — a self-hosted AI workspace with memory, tools, voice, and research.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CAMELOT WORKSPACE (:7000)                   │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │  FastAPI     │  │  Agent Loop  │  │  System Prompt         ││
│  │  app.py      │──│  src/        │──│  SOUL.md + tools       ││
│  │  routes/     │  │  agent_loop  │  │  + memory context      ││
│  └──────┬──────┘  └──────┬───────┘  └────────────────────────┘│
│         │                │                                      │
│  ┌──────┴──────┐  ┌─────┴────────────────────────────────┐    │
│  │  PWA UI     │  │  Tool Parsing Pipeline                │    │
│  │  Camelot    │  │  1. Native function calls (primary)   │    │
│  │  theme      │  │  2. Fenced blocks (secondary)         │    │
│  └─────────────┘  │  3. Bracket markers *[...] (tertiary) │    │
│                    │  4. XML/DSML/TOOL_CALL (fallbacks)    │    │
│                    └──────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MEMORY                                                  │   │
│  │  Tier 1 (session): ChromaDB + JSON  <-> manage_memory    │   │
│  │  Tier 2 (vault):   vault_recall     <-> VaultRecallProv  │   │
│  │  Promotion:         meditation cron  (T1 -> T2)          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SERVICES                                                │   │
│  │  llama-server  :8080  (Guinevere inference)              │   │
│  │  Fish Speech   :7300  (voice TTS)                        │   │
│  │  Ollama        :11434 (embeddings, nomic-embed-text)     │   │
│  │  ChromaDB      :8100  (Docker, session memory vectors)   │   │
│  │  SearXNG       :8889  (Docker, web search)               │   │
│  │  MCP servers          (email, Plaid, filesystem)         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Start everything (if not using systemd)
docker compose -f docker-compose.camelot.yml up -d   # ChromaDB + SearXNG
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 7000

# Or via systemd (recommended)
sudo systemctl start camelot-workspace
```

Open http://localhost:7000 and log in. Add Guinevere as a model endpoint
in Settings: `http://localhost:8080/v1`, model name `guinevere-r3-q5_k_m`.
Select the "Guinevere" preset, and start talking.

---

## Port Map

| Service | Port | Protocol | Managed By |
|---------|------|----------|------------|
| Camelot Workspace | 7000 | HTTP | systemd |
| Guinevere (llama-server) | 8080 | HTTP (OpenAI-compat) | separate service |
| Fish Speech TTS | 7300 | HTTP | separate service |
| Ollama | 11434 | HTTP | separate service |
| ChromaDB | 8100 | HTTP | Docker (docker-compose.camelot.yml) |
| SearXNG | 8889 | HTTP | Docker (docker-compose.camelot.yml) |

---

## How the Agent Loop Works

When you send a message, this is the full pipeline:

### 1. System Prompt Assembly
- **SOUL.md** (Guinevere's identity) injected as the first system message via the preset system
- **Tool preamble** added by `_build_system_prompt()` — compact mode for API models
- **Memory context** injected by `build_context_preface()`:
  - Tier 1: ChromaDB native memory (session facts, pinned preferences)
  - Tier 2: vault_recall (Obsidian vault long-term knowledge)
- **MCP tool descriptions** injected for email, filesystem, etc.
- **Current date/time** prepended

### 2. Tool Parsing Pipeline
Guinevere's response is parsed for tool invocations in this priority order:

1. **Native function calls** (primary) — OpenAI-format `tool_calls` in the response. Qwen 3.6 supports this natively. Enabled because `guinevere` is in the `_model_supports_tools` keyword list and `localhost` is in `_API_HOSTS`.

2. **Fenced code blocks** (secondary) — ` ```bash`, ` ```python`, ` ```web_search` etc. Standard Odysseus format for local models.

3. **Bracket markers** (tertiary) — `*[Searching memory -- query]*` style markers from Guinevere's fine-tuned training data. Custom parser added in `tool_parsing.py`.

4. **XML/DSML/TOOL_CALL** (fallbacks) — Various model-specific formats (DeepSeek DSML, MiniMax tool_code, etc.)

### 3. Agent State Machine
- Max **20 rounds** per task (configurable)
- **Stall detection**: 4 consecutive useless rounds triggers force-answer
- **Runaway detection**: same tool called 15+ times triggers force-answer
- **Self-termination**: agent declares DONE, BLOCKED, or continues
- **Context compaction**: auto-triggers at 85% of context window, LLM self-summarizes older turns

---

## Memory Architecture

### Tier 1: Session Memory (ChromaDB + JSON)
- **What**: Facts, preferences, decisions from current and recent sessions
- **Storage**: ChromaDB vectors (cosine similarity) + JSON file (`data/memory.json`)
- **Embeddings**: fastembed ONNX (sentence-transformers/all-MiniLM-L6-v2), runs on CPU
- **Tool**: `manage_memory` (add/search/edit/delete/list)
- **Auto-injection**: Top 3 relevant memories injected into every turn via `build_context_preface()`

### Tier 2: Vault Recall (sqlite-vec + Obsidian)
- **What**: Deep long-term knowledge — identity, project history, meditation extracts, cross-session context
- **Storage**: `~/.hermes/vault_recall.db` (sqlite-vec), indexed from `~/obsidian-vault/Camelot V3/memory/`
- **Embeddings**: nomic-embed-text via Ollama on :11434
- **Tool**: `vault_search` (query the vault directly)
- **Auto-injection**: Top 3 vault results injected alongside session memory
- **Read-only**: Writes go through the meditation promotion path, not through the workspace UI

### Promotion Path (Tier 1 -> Tier 2)
The meditation cron cycle consolidates important session memories into Obsidian vault extracts. This pipeline is external to Camelot Workspace and runs independently.

---

## Voice (Fish Speech TTS)

- **Provider**: Fish Speech 1.5 BASE on :7300
- **Voice**: `gwen_ref` (Guinevere's cloned voice reference)
- **API**: `POST /api/tts` with form data: `text=...&reference=gwen_ref`
- **Format**: Returns WAV audio (16-bit PCM, mono, 44100 Hz)
- **Config**: `data/settings.json` — `tts_provider: fish_speech`
- **Cache**: `data/tts_cache/` — SHA256-keyed audio files

To change voice settings, edit `data/settings.json` or use the Settings UI (admin).

---

## MCP Servers

### Registered Servers

| Server | ID | Transport | Tools |
|--------|-----|-----------|-------|
| Camelot Email | camelot-email | stdio | 7 (list, read, send, search, reply, accounts, sms) |
| Built-in: Email | email | stdio | 11 (IMAP/SMTP operations) |
| Built-in: Memory | memory | stdio | 1 (manage_memory) |
| Built-in: RAG | rag | stdio | 1 (document search) |
| Built-in: Image Gen | image_gen | stdio | 1 (generate_image) |

### Adding New MCP Servers
Register via the Settings UI (MCP tab) or insert directly into the database:

```python
from core.database import SessionLocal, McpServer
import json

db = SessionLocal()
srv = McpServer(
    id='my-server',
    name='My MCP Server',
    transport='stdio',
    command='/usr/bin/python3',
    args=json.dumps(['/path/to/mcp_server.py']),
    env=json.dumps({}),
)
db.add(srv)
db.commit()
db.close()
```

Restart the workspace to connect the new server.

---

## Preset System

Presets inject a system prompt that shapes Guinevere's personality and behavior.

| Preset | Temperature | Purpose |
|--------|-------------|---------|
| **guinevere** (default) | 0.7 | SOUL.md — full Guinevere personality |
| code_analyze | 0.2 | Code analysis mode |
| brainstorm | 0.9 | Creative ideation |
| reason | 0.3 | Systematic reasoning |
| custom | 1.0 | User-defined |

The Guinevere preset loads `~/.hermes/SOUL-GWEN.md` at startup. Edit that file to change her personality. Changes take effect on next app restart.

---

## Theming

The Camelot theme is the default. 22 additional themes are available in the Settings UI.

### Camelot Theme Colors
| Element | Color | Hex |
|---------|-------|-----|
| Background | Deep royal purple-black | `#1a1625` |
| Foreground | Warm parchment | `#e8d5b7` |
| Panel | Darker purple | `#0f0d18` |
| Border | Muted purple | `#3d2e5c` |
| Accent (gold) | Royal gold | `#c9a84c` |
| User bubble | Purple | `#2a2040` |
| AI bubble | Deep purple | `#1e1830` |

Background pattern: constellations (gold, 40% intensity).

---

## Cookbook (Model Management)

The Cookbook is Odysseus's hardware-aware model recommender. It scans your GPU (3090 Ti) and scores models on:

- **Quality** (45%): Parameter count + family bonuses + quantization penalties
- **Speed** (30%): Estimated tok/s vs use-case target
- **Fit** (15%): VRAM utilization ratio (sweet spot: 50-80%)
- **Context** (10%): Available context window

Useful for managing Guinevere and PJ (Prometheus Jones) model swaps, and for evaluating new models before downloading.

---

## Systemd Management

```bash
# Service commands
sudo systemctl start camelot-workspace
sudo systemctl stop camelot-workspace
sudo systemctl restart camelot-workspace
sudo systemctl status camelot-workspace

# View logs
journalctl -u camelot-workspace -f              # live tail
journalctl -u camelot-workspace --since "1h ago" # last hour
journalctl -u camelot-workspace -n 50            # last 50 lines

# Docker services (managed by ExecStartPre)
docker compose -f docker-compose.camelot.yml ps
docker compose -f docker-compose.camelot.yml logs chromadb
docker compose -f docker-compose.camelot.yml logs searxng
```

The service auto-starts Docker services (ChromaDB + SearXNG) before launching uvicorn.

---

## Upstream Sync

Camelot Workspace tracks upstream Odysseus via git remote:

```bash
git fetch upstream
git log upstream/main --oneline -10          # see what's new
git cherry-pick <commit-hash>                # pick individual changes
```

All intentional divergences from upstream are documented in `CAMELOT_DIFF.md`.
Check that file before merging to avoid conflicts in modified files.

---

## File Map (Modified/New)

### Modified from Upstream
| File | Change |
|------|--------|
| `app.py` | Branding, vault_recall init, memory_provider_registry extraction |
| `core/constants.py` | APP_VERSION |
| `src/agent_loop.py` | `guinevere` in model keywords |
| `src/tool_parsing.py` | Bracket marker parser (Pattern 6) |
| `src/chat_processor.py` | vault_recall Tier 2 injection |
| `src/app_initializer.py` | VaultRecallProvider registration |
| `src/preset_manager.py` | Guinevere preset with SOUL.md |
| `services/tts/tts_service.py` | Fish Speech provider |
| `static/js/theme.js` | Camelot theme |
| `static/index.html` | Title, favicon, branding |
| `static/manifest.json` | PWA branding |

### New Files
| File | Purpose |
|------|---------|
| `src/vault_recall_provider.py` | Tier 2 memory provider (Obsidian vault) |
| `docker-compose.camelot.yml` | ChromaDB + SearXNG (no main app) |
| `camelot-workspace.service` | systemd unit |
| `scripts/start-services.sh` | Docker pre-start script |
| `CAMELOT_DIFF.md` | Upstream divergence tracking |
| `CAMELOT_GUIDE.md` | This file |

---

## Troubleshooting

### App won't start
```bash
# Check if port 7000 is taken
ss -tlnp | grep 7000
# Check Python imports
cd ~/camelot-workspace && source venv/bin/activate && python -c "import app"
# Check Docker services
docker compose -f docker-compose.camelot.yml ps
```

### Guinevere not responding
```bash
# Check llama-server
curl -s http://localhost:8080/v1/models
# Verify model endpoint is registered in Settings UI
# Check agent logs
journalctl -u camelot-workspace | grep "agent\|error" | tail -20
```

### vault_recall not connecting
```bash
# Check DB exists
ls -la ~/.hermes/vault_recall.db
# Check Ollama for embeddings
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; [print(m['name']) for m in json.load(sys.stdin)['models']]"
# Check sqlite-vec installed
source venv/bin/activate && python -c "import sqlite_vec; print('OK')"
```

### Voice not working
```bash
# Check Fish Speech
curl -s http://localhost:7300/api/health
# Test synthesis
curl -s -X POST http://localhost:7300/api/tts -d "text=test&reference=gwen_ref" -o /tmp/test.wav
file /tmp/test.wav  # should say RIFF WAVE audio
# Check settings
cat data/settings.json | python3 -m json.tool
```

### ChromaDB connection refused
```bash
docker compose -f docker-compose.camelot.yml up -d chromadb
curl -s http://localhost:8100/api/v2/heartbeat
```
