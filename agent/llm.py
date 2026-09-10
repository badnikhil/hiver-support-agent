"""Thin Ollama chat wrapper with an on-disk cache."""
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
CACHE = Path(__file__).resolve().parent.parent / "data/cache/llm.sqlite"
CACHE.parent.mkdir(parents=True, exist_ok=True)
_db = sqlite3.connect(CACHE)
_db.execute("create table if not exists cache (k text primary key, v text)")


def chat(system, user, *, json_mode=False, temperature=0, max_tokens=200, model=None):
    body = {
        "model": model or MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        # fixed num_ctx: changing it reloads the model (+5s); same system prompt => KV cache hit
        "options": {"temperature": temperature, "num_ctx": 4096, "num_predict": max_tokens},
    }
    if json_mode:
        body["format"] = "json"
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    row = _db.execute("select v from cache where k=?", (key,)).fetchone()
    if row:
        return row[0]
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=120)
    r.raise_for_status()
    out = r.json()["message"]["content"]
    _db.execute("insert or replace into cache values (?,?)", (key, out))
    _db.commit()
    return out


if __name__ == "__main__":
    import time
    t = time.time()
    print(chat("Answer in one word.", "Capital of France?"), f"{time.time()-t:.2f}s")
    t = time.time()
    print(chat("Answer in one word.", "Capital of France?"), f"cached {time.time()-t:.3f}s")
