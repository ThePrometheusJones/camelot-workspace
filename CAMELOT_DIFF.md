# Camelot Workspace — Divergences from Upstream Odysseus

Forked from `pewdiepie-archdaemon/odysseus` (commit at fork time: 2026-06-04).
Upstream remote: `upstream` → `https://github.com/pewdiepie-archdaemon/odysseus.git`

When cherry-picking upstream updates, check these files for conflicts.

## Modified Files

### Core Identity
| File | Change |
|------|--------|
| `core/constants.py` | APP_VERSION → "1.0.0-camelot" |
| `app.py` | FastAPI title/description → "Camelot Workspace"; extract memory_provider_registry; vault_recall init in startup |
| `src/preset_manager.py` | Added "guinevere" preset loading SOUL-GWEN.md; set as first DEFAULT_PRESET |

### Agent Loop
| File | Change |
|------|--------|
| `src/agent_loop.py` | Added "guinevere" to `_model_supports_tools` keyword list (line ~1520) |
| `src/tool_parsing.py` | Added Pattern 6: bracket marker `*[Action — query]*` regex + `_BRACKET_ACTION_MAP` + `_parse_bracket_markers()` |

### Memory
| File | Change |
|------|--------|
| `src/app_initializer.py` | Import + register VaultRecallProvider; pass memory_provider_registry to ChatProcessor |
| `src/chat_processor.py` | Added `memory_provider_registry` param to __init__; vault_recall Tier 2 context injection in `build_context_preface()` |

### Voice
| File | Change |
|------|--------|
| `services/tts/tts_service.py` | Added `fish_speech` provider type calling :7300; `_synthesize_fish_speech()` method |

### UI
| File | Change |
|------|--------|
| `static/js/theme.js` | Added "camelot" theme (first in THEMES); DEFAULT_THEME → "camelot"; pattern/effect/intensity defaults |
| `static/index.html` | Title → "Camelot Workspace"; favicon → gold star SVG; all "Odysseus" display text → "Camelot" |
| `static/manifest.json` | name/short_name → "Camelot Workspace"/"Camelot"; colors updated |

## New Files

| File | Purpose |
|------|---------|
| `src/vault_recall_provider.py` | MemoryProvider wrapping vault_recall.db (sqlite-vec + nomic-embed-text) |
| `docker-compose.camelot.yml` | Minimal compose: ChromaDB :8100 + SearXNG :8889 |
| `camelot-workspace.service` | systemd unit file |
| `scripts/start-services.sh` | Pre-start script for Docker services |
| `CAMELOT_DIFF.md` | This file |

## Configuration
| File | Notes |
|------|-------|
| `.env` | Camelot-specific: SearXNG on :8889, ChromaDB on :8100, LOCALHOST_BYPASS=true |
| `data/settings.json` | tts_provider=fish_speech, voice=gwen_ref |
| `data/presets.json` | Generated at runtime with "guinevere" preset |

## DB Records
| Table | Record |
|-------|--------|
| McpServer | `camelot-email` — stdio, /usr/bin/python3, /home/ken/camelot-email/mcp_server.py |

## Upstream Merge Strategy

1. `git fetch upstream`
2. `git log upstream/main --oneline -20` to review changes
3. Cherry-pick individual commits: `git cherry-pick <hash>`
4. If conflict in a modified file above, manually resolve preserving our changes
5. Test after merge: `curl http://localhost:7000/api/tts/stats` + `curl http://localhost:7000/api/presets`
