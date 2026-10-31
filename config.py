"""Settings shared by every script: read from .env next to this file, then overridden by the environment.

See .env.example for every key and its default.
"""

import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_env(path=ROOT / ".env"):
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.split(" #", 1)[0].strip().strip('"').strip("'")
    return {**env, **os.environ}


ENV = load_env()


def get(key, default=None):
    v = ENV.get(key)
    return v if v not in (None, "") else default


def vault_path():
    return (ROOT / get("VAULT_PATH", "sample_vault")).expanduser().resolve()


def database():
    return get("NEO4J_DATABASE", "neo4j")


def driver():
    from neo4j import GraphDatabase
    if not get("NEO4J_PASSWORD"):
        raise SystemExit("NEO4J_PASSWORD is empty: copy .env.example to .env and set it")
    return GraphDatabase.driver(get("NEO4J_URI", "bolt://localhost:7687"),
                                auth=(get("NEO4J_USER", "neo4j"), get("NEO4J_PASSWORD")),
                                notifications_min_severity="OFF")


def scope_prefix(folder):
    """'' (whole vault) stays ''; 'Projects/X' becomes 'Projects/X/' so a STARTS WITH filter matches only that folder."""
    folder = (folder or "").strip("/")
    return folder + "/" if folder else ""


# The note author: notes say "I" and "me", and the extractor and resolver map those to this person.
AUTHOR_NAME = get("AUTHOR_NAME", "the author")
AUTHOR_ALIASES = {a.strip().lower() for a in [AUTHOR_NAME, AUTHOR_NAME.split()[0], "i", "me"] if a.strip()}


# ---------- embeddings ----------
# ollama:    bge-m3 through a local Ollama server (1,024 dims, multilingual). What the published eval used.
# fastembed: BAAI/bge-small-en-v1.5 on CPU through ONNX (384 dims, English). A small download for the quickstart.

EMB_BACKEND = get("EMBED_BACKEND", "ollama")
EMB_MODEL = get("EMBED_MODEL", "bge-m3" if EMB_BACKEND == "ollama" else "BAAI/bge-small-en-v1.5")
EMB_DIMS = int(get("EMBED_DIMS", 1024 if EMB_BACKEND == "ollama" else 384))
OLLAMA_URL = get("OLLAMA_URL", "http://localhost:11434") + "/api/embed"
_fastembed = None


def embed_many(texts):
    global _fastembed
    if not texts:
        return []
    if EMB_BACKEND == "ollama":
        req = urllib.request.Request(OLLAMA_URL, data=json.dumps({"model": EMB_MODEL, "input": texts}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["embeddings"]
    if EMB_BACKEND == "fastembed":
        if _fastembed is None:
            from fastembed import TextEmbedding
            local = get("EMBED_MODEL_PATH")  # a pre-downloaded model folder, for offline machines
            _fastembed = TextEmbedding(model_name=EMB_MODEL, **({"specific_model_path": local} if local else {}))
        return [v.tolist() for v in _fastembed.embed(list(texts))]
    raise SystemExit(f"Unknown EMBED_BACKEND {EMB_BACKEND!r}: use ollama or fastembed")


def embed(text):
    return embed_many([text])[0]
