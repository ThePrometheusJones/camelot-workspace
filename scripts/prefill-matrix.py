#!/usr/bin/env python3
"""
Prefill benchmark matrix for V6 (MoE 35B-A3B).

Per cell: start llama-server → poll /health → record load time →
send the real Guinevere cold prefix → record prefill tok/s, TTFT,
gen tok/s (with MTP), peak VRAM → kill.

Usage:
    # Capture the cold prefix first (while Bailey is running):
    python3 scripts/prefill-matrix.py --capture-prompt

    # Run the coarse matrix (Guinevere goes offline):
    python3 scripts/prefill-matrix.py --run

    # Restore the current config after:
    python3 scripts/prefill-matrix.py --restore
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests

# ── Config ──────────────────────────────────────────────────────────── #

LLAMA_BIN = os.path.expanduser(
    "~/camelot-inference/llama-cpp-mainline/build/bin/llama-server"
)
GGUF_PATH = "/mnt/storage/camelot-inference/models/guinevere-v6-q5_k_m.gguf"
PORT = 8090  # Use a different port so we don't collide with production
PROMPT_CACHE = os.path.join(os.path.dirname(__file__), ".prefill-prompt.json")
BAILEY_URL = os.environ.get("BAILEY_URL", "http://localhost:7000")
LLAMA_URL = f"http://localhost:{PORT}"

# Fixed flags across all cells
BASE_ARGS = [
    "-ngl", "99",
    "--flash-attn", "on",
    "-np", "1",
    "--no-mmap", "--mlock",
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--spec-type", "draft-mtp",
    "--spec-draft-n-max", "2",
]

# Coarse matrix: ncmoe × ctx, q8_0 KV
COARSE_MATRIX = [
    {"ncmoe": 20, "ctx": 262144, "kv": "q8_0"},
    {"ncmoe": 20, "ctx": 64000,  "kv": "q8_0"},
    {"ncmoe": 10, "ctx": 262144, "kv": "q8_0"},
    {"ncmoe": 10, "ctx": 64000,  "kv": "q8_0"},
    {"ncmoe":  0, "ctx": 262144, "kv": "q8_0"},
    {"ncmoe":  0, "ctx": 64000,  "kv": "q8_0"},
]

# Reference row: f16 KV, 64K, ncmoe 0
REFERENCE_ROW = {"ncmoe": 0, "ctx": 64000, "kv": "f16"}

USER_MSG = "Show me the current UI theme setting."


# ── Helpers ─────────────────────────────────────────────────────────── #

def vram_mib():
    """Return (used_mib, total_mib) from nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, total = [int(x.strip()) for x in out.split(",")]
        return used, total
    except Exception:
        return 0, 0


def kill_port_8090():
    """Kill any llama-server on the benchmark port."""
    try:
        pids = subprocess.check_output(
            ["pgrep", "-f", f"llama-server.*--port {PORT}"],
            text=True,
        ).strip().split()
        for pid in pids:
            os.kill(int(pid), signal.SIGTERM)
        time.sleep(3)
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(2)
    except subprocess.CalledProcessError:
        pass


def wait_health(timeout=180):
    """Poll /health until ready. Returns seconds waited or -1."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{LLAMA_URL}/health", timeout=3)
            if r.ok:
                return round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(2)
    return -1


def build_cell_args(cell):
    """Build llama-server args for one matrix cell."""
    args = list(BASE_ARGS)
    ncmoe = cell["ncmoe"]
    if ncmoe > 0:
        args += ["-ncmoe", str(ncmoe)]
    kv = cell["kv"]
    args += ["--cache-type-k", kv, "--cache-type-v", kv]
    args += ["-c", str(cell["ctx"])]
    return args


def run_cell(cell, messages, tools):
    """Start server, measure, kill. Returns dict with metrics or error."""
    label = f"ncmoe={cell['ncmoe']} ctx={cell['ctx']//1024}K kv={cell['kv']}"
    print(f"\n{'='*60}")
    print(f"  Cell: {label}")
    print(f"{'='*60}")

    kill_port_8090()

    # Check VRAM headroom
    used_before, total = vram_mib()
    print(f"  VRAM before: {used_before} / {total} MiB")

    args = [LLAMA_BIN, "-m", GGUF_PATH] + build_cell_args(cell)
    print(f"  Args: {' '.join(args[3:])}")

    # Launch
    log = open(f"/tmp/prefill-matrix-{cell['ncmoe']}-{cell['ctx']}-{cell['kv']}.log", "w")
    env = dict(os.environ, HIP_VISIBLE_DEVICES="-1", ROCR_VISIBLE_DEVICES="-1")
    proc = subprocess.Popen(args, stdout=log, stderr=log, env=env)
    print(f"  PID: {proc.pid}")

    load_time = wait_health(timeout=240)
    if load_time < 0:
        print(f"  SKIP: server failed to start")
        proc.kill()
        proc.wait()
        log.close()
        return {"label": label, "error": "failed to start"}

    used_loaded, _ = vram_mib()
    vram_model = used_loaded - used_before
    print(f"  Load time: {load_time}s")
    print(f"  VRAM after load: {used_loaded} MiB (+{vram_model} MiB)")

    # Send the cold prompt
    payload = {
        "model": "benchmark",
        "messages": messages,
        "tools": tools,
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": False,
    }

    try:
        t0 = time.time()
        r = requests.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json=payload,
            timeout=600,
        )
        wall = time.time() - t0
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        print(f"  SKIP: completion failed: {e}")
        proc.kill()
        proc.wait()
        log.close()
        return {"label": label, "error": str(e)}

    used_peak, _ = vram_mib()

    usage = resp.get("usage", {})
    timings = resp.get("timings", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    prefill_tps = timings.get("prompt_per_second", 0)
    gen_tps = timings.get("predicted_per_second", 0)
    ttft = timings.get("prompt_ms", 0) / 1000.0 if timings.get("prompt_ms") else wall

    result = {
        "label": label,
        "ncmoe": cell["ncmoe"],
        "ctx": cell["ctx"],
        "kv": cell["kv"],
        "load_time_s": load_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefill_tps": round(prefill_tps, 1),
        "gen_tps": round(gen_tps, 1),
        "ttft_s": round(ttft, 2),
        "wall_s": round(wall, 2),
        "vram_model_mib": vram_model,
        "vram_peak_mib": used_peak,
    }

    print(f"  prompt_tokens: {prompt_tokens}")
    print(f"  prefill: {result['prefill_tps']} tok/s")
    print(f"  gen (MTP): {result['gen_tps']} tok/s")
    print(f"  TTFT: {result['ttft_s']}s")
    print(f"  VRAM peak: {used_peak} MiB")

    # Kill
    proc.terminate()
    proc.wait()
    log.close()
    time.sleep(3)

    return result


# ── Prompt capture ──────────────────────────────────────────────────── #

def capture_prompt():
    """Capture the real Guinevere cold prefix from live Bailey."""
    print("Capturing cold prefix from Bailey...")

    # Import Bailey modules — must run from Bailey's venv for full imports.
    # Import agent_loop FIRST to resolve the circular dep between
    # tool_schemas and agent_tools.
    sys.path.insert(0, os.path.expanduser("~/camelot-workspace"))
    os.environ.setdefault("DATABASE_URL", "sqlite:///data/app.db")

    import importlib
    try:
        al_mod = importlib.import_module("src.agent_loop")
        system_prompt = al_mod.AGENT_SYSTEM_PROMPT
        mod = importlib.import_module("src.tool_schemas")
        FUNCTION_TOOL_SCHEMAS = mod.FUNCTION_TOOL_SCHEMAS
    except Exception as e:
        print(f"  FATAL: Bailey import failed: {e}")
        print("  Run from Bailey's venv: ./venv/bin/python3 scripts/prefill-matrix.py --capture-prompt")
        sys.exit(1)

    # Get MCP tools from live Bailey
    try:
        r = requests.get(f"{BAILEY_URL}/api/mcp/tools", timeout=10)
        r.raise_for_status()
        mcp_tools_raw = r.json()
    except Exception as e:
        print(f"  WARNING: Could not fetch MCP tools: {e}")
        mcp_tools_raw = []

    # Build MCP tool schemas in OpenAI format
    mcp_schemas = []
    for t in mcp_tools_raw:
        schema = {
            "type": "function",
            "function": {
                "name": t["qualified_name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
            }
        }
        mcp_schemas.append(schema)

    all_tools = list(FUNCTION_TOOL_SCHEMAS) + mcp_schemas
    print(f"  Tools: {len(FUNCTION_TOOL_SCHEMAS)} builtin + {len(mcp_schemas)} MCP = {len(all_tools)}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_MSG},
    ]

    payload = {"messages": messages, "tools": all_tools}

    with open(PROMPT_CACHE, "w") as f:
        json.dump(payload, f)

    # Rough token estimate
    total_chars = sum(len(json.dumps(m)) for m in messages) + len(json.dumps(all_tools))
    est_tokens = total_chars // 4
    print(f"  System prompt: {len(system_prompt)} chars")
    print(f"  Total payload: {total_chars} chars (~{est_tokens} tokens)")
    print(f"  Saved to: {PROMPT_CACHE}")


# ── Main ────────────────────────────────────────────────────────────── #

def print_table(results):
    """Print results as a formatted table."""
    print(f"\n{'='*100}")
    print("PREFILL MATRIX RESULTS")
    print(f"{'='*100}")
    header = f"{'ncmoe':>6} {'ctx':>6} {'kv':>5} {'load_s':>7} {'prompt':>8} {'prefill':>9} {'gen':>7} {'ttft_s':>7} {'vram_mib':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r.get('ncmoe','?'):>6} {r.get('ctx','?'):>6} {r.get('kv','?'):>5}   SKIP: {r['error']}")
            continue
        print(
            f"{r['ncmoe']:>6} "
            f"{r['ctx']//1024:>5}K "
            f"{r['kv']:>5} "
            f"{r['load_time_s']:>7.1f} "
            f"{r['prompt_tokens']:>8} "
            f"{r['prefill_tps']:>8.1f} "
            f"{r['gen_tps']:>6.1f} "
            f"{r['ttft_s']:>7.2f} "
            f"{r['vram_peak_mib']:>9}"
        )
    print(f"{'='*100}")


def restore_production():
    """Restore the production V6 config via systemd."""
    print("\nRestoring production V6...")
    kill_port_8090()
    subprocess.run(["sudo", "systemctl", "start", "guinevere.service"], check=False)
    # Wait for production health
    prod_url = "http://localhost:8080"
    for _ in range(60):
        try:
            r = requests.get(f"{prod_url}/health", timeout=3)
            if r.ok:
                print("  Guinevere restored on :8080")
                return True
        except Exception:
            pass
        time.sleep(3)
    print("  WARNING: Guinevere did not come back healthy")
    return False


def main():
    parser = argparse.ArgumentParser(description="V6 prefill benchmark matrix")
    parser.add_argument("--capture-prompt", action="store_true",
                        help="Capture the real cold prefix from Bailey (run first)")
    parser.add_argument("--run", action="store_true",
                        help="Run the coarse benchmark matrix")
    parser.add_argument("--restore", action="store_true",
                        help="Restore production V6 config")
    parser.add_argument("--include-reference", action="store_true",
                        help="Include f16 KV reference row")
    args = parser.parse_args()

    if args.capture_prompt:
        capture_prompt()
        return

    if args.restore:
        restore_production()
        return

    if args.run:
        if not os.path.exists(PROMPT_CACHE):
            print("ERROR: Run --capture-prompt first to capture the cold prefix.")
            sys.exit(1)

        with open(PROMPT_CACHE) as f:
            payload = json.load(f)
        messages = payload["messages"]
        tools = payload["tools"]
        print(f"Loaded prompt: {len(tools)} tools")

        # Stop production Guinevere
        print("\nStopping production Guinevere...")
        subprocess.run(["sudo", "systemctl", "stop", "guinevere.service"], check=False)
        # Also kill any orphaned llama-server on :8080
        try:
            pids = subprocess.check_output(
                ["pgrep", "-f", "llama-server.*--port 8080"],
                text=True,
            ).strip().split()
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
            time.sleep(3)
        except (subprocess.CalledProcessError, ProcessLookupError):
            pass

        time.sleep(5)

        results = []
        for cell in COARSE_MATRIX:
            r = run_cell(cell, messages, tools)
            results.append(r)

        if args.include_reference:
            r = run_cell(REFERENCE_ROW, messages, tools)
            results.append(r)

        print_table(results)

        # Save raw results
        out_file = os.path.expanduser("~/prefill-matrix-results.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nRaw results saved to: {out_file}")

        # Restore
        print("\nRestoring production config...")
        restore_production()

    if not any([args.capture_prompt, args.run, args.restore]):
        parser.print_help()


if __name__ == "__main__":
    main()
