"""Graph method: `graph_search.py ask`, run as a subprocess exactly as an agent would call it.

The hybrid base list, then entry entities (named in the question + seeds from the top chunks) -> 1-hop typed links
that never pass through hubs -> dated claims -> chunks only the graph reached, appended after the hybrid ones.

Chunks are returned, not notes, so this asks for 3x as many chunks as notes are scored (run_eval collapses them to
notes at their best rank) and lifts the token budget so the ranked tail isn't cut by text length. The claim and
link lines go into `context`, which run_eval counts against the same 8,000-token budget (40% share).
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "graph_search.py"


def retrieve(question, scope, k, extra=()):
    p = subprocess.run([sys.executable, str(SCRIPT), "ask", question, "--scope", scope or "",
                        "--k", str(k * 3), "--budget", "200000", *extra],
                       capture_output=True, text=True, cwd=SCRIPT.parent)
    if p.returncode != 0:
        err = next((ln for ln in p.stderr.splitlines() if ln.startswith("GRAPH_ERROR")), p.stderr.strip()[-300:])
        return {"ranked": [], "context": [], "error": err}
    out = json.loads(p.stdout)
    ranked = [{"path": r["path"], "chunk_id": r["chunk_id"], "text": r["text"]} for r in out["ranked"]]
    return {"ranked": ranked, "context": out["context"], "error": None}
