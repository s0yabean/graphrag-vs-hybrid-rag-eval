"""Entity resolution, claim timelines and the graph_search helpers, on the hand-written sample extraction.

load_graph reads chunk ids from Neo4j; here they come from build_chunks.build_rows instead, so everything up to
the Cypher writes runs without a database.
"""

import config
import graph_search
import load_graph
import pytest


@pytest.fixture(scope="module")
def graph_layer(sample_rows):
    def chunk_rows(folder):
        prefix = config.scope_prefix(folder)
        return [{"id": r["id"], "note_date": r["note_date"], "path": r["note"]}
                for r in sample_rows if r["note"].startswith(prefix)]

    mp = pytest.MonkeyPatch()
    mp.setattr(load_graph, "chunk_rows", chunk_rows)
    docs, missing = load_graph.load_docs([""], "sample")
    mp.undo()
    canon, _, _ = load_graph.resolve(docs, [])
    ents, _, rels, claims, skipped, _ = load_graph.build(docs, canon)
    superseded, conflicts = load_graph.timelines(claims)
    by_name = {e["name"]: e for e in ents.values()}
    return {"docs": docs, "missing": missing, "ents": ents, "by_name": by_name, "rels": rels, "claims": claims,
            "superseded": superseded, "conflicts": conflicts, "skipped": skipped}


def test_every_sample_chunk_loads(graph_layer):
    assert len(graph_layer["docs"]) == 54 and graph_layer["missing"] == 0
    assert graph_layer["skipped"] == 0                      # every relationship endpoint resolved


def test_aliases_merge_into_one_entity(graph_layer):
    fernwick = graph_layer["by_name"]["Fernwick Roasting Co"]
    assert {"FRC", "Fernwick"} <= fernwick["aliases"]
    assert "Fernwick at Home" in graph_layer["by_name"]    # a different entity, not folded in
    # The two Sams stay apart: a bare first name is never a merging alias.
    assert {"Sam Patel", "Sam Whitlock"} <= graph_layer["by_name"].keys()
    assert "Sam" not in graph_layer["by_name"]


def test_harbour_blend_price_timeline(graph_layer):
    hb = graph_layer["by_name"]["Harbour Blend"]["id"]
    prices = sorted((c for c in graph_layer["claims"] if c["about"] == hb and c["attr_key"] == "wholesale price"),
                    key=lambda c: c["note_date"])
    current = [c for c in prices if c["is_current"]]
    assert [c["value"] for c in current] == ["$28.00 per kg"]
    # Plans and the AI chat's suggestion are kept, but never take part in supersession.
    ai = next(c for c in prices if c["modality"] == "ai_suggestion")
    assert ai["value"] == "$32 per kg" and ai["is_current"] is None
    assert all(c["is_current"] is None for c in prices if c["modality"] == "plan")

    by_id = {c["id"]: c for c in graph_layer["claims"]}
    chain = [(by_id[s["a"]]["value"], by_id[s["b"]]["value"]) for s in graph_layer["superseded"]
             if by_id[s["a"]]["about"] == hb and by_id[s["a"]]["attr_key"] == "wholesale price"]
    assert chain == [("$24.00 per kg", "$26.50 per kg"), ("$26.50 per kg", "$28.00 per kg")]


def test_led_to_chain_from_late_shipments_to_lost_client(graph_layer):
    """The README's `path` example: late shipments -> supplier switch -> price rise -> lost client."""
    names = {i: e["name"] for i, e in graph_layer["ents"].items()}
    led = {}
    for s, typ, t in graph_layer["rels"]:
        if typ == "LED_TO":
            led.setdefault(names[s], []).append(names[t])
    paths, frontier = {}, [["Late Alta Verde shipments 2025"]]
    for _ in range(3):
        frontier = [p + [n] for p in frontier for n in led.get(p[-1], [])]
        paths.update({p[-1]: p for p in frontier})
    assert paths["The Copper Kettle"] == ["Late Alta Verde shipments 2025",
                                          "Switch green coffee supplier to Cooperativa Rio Claro",
                                          "Raise Harbour Blend to $28.00 per kg", "The Copper Kettle"]


def test_timelines_rules_on_toy_claims():
    def claim(i, path, date, value, modality="fact", attr="price"):
        return {"id": i, "about": "e", "attr_key": attr, "path": path, "note_date": date, "valid_from": None,
                "value": value, "modality": modality}
    claims = [claim("old", "a.md", "2024-01-01", "$24"), claim("new", "b.md", "2025-01-01", "$26"),
              claim("same", "c.md", "2025-01-01", "$27"),                       # same date, different value
              claim("dup", "d.md", "2025-01-01", "26 dollars"),                 # same number: not a change
              claim("plan", "e.md", "2026-01-01", "$30", modality="plan"),
              claim("f1", "f.md", "2024-01-01", "fast", attr="findings"),       # list attribute: no timeline
              claim("f2", "g.md", "2025-01-01", "cheap", attr="findings")]
    superseded, conflicts = load_graph.timelines(claims)
    assert {"a": "old", "b": "new"} in superseded
    assert conflicts and all({c["a"], c["b"]} & {"same"} for c in conflicts)
    status = {c["id"]: c["is_current"] for c in claims}
    assert status["old"] is False and status["new"] is True and status["plan"] is None
    assert status["f1"] is None and status["f2"] is None


def test_name_normalisation_and_value_matching():
    assert load_graph.norm("The Copper Kettle Pte. Ltd.") == "copper kettle"
    assert load_graph.norm("Tallybook.io") == "tallybook"
    assert load_graph.same_value("3,000+", "over 3000")
    assert load_graph.same_value("15-50%", "15–50% uplift")
    assert not load_graph.same_value("$26.50", "$28.00")
    assert load_graph.key_of("I", "Person") == load_graph.key_of("Ruth Calloway", "Person")
    assert load_graph.specific_alias("Sam Patel", "Person") and not load_graph.specific_alias("Sam", "Person")


def test_resolve_ignores_alias_that_is_another_entitys_name():
    docs = [{"entities": [{"name": "Fernwick", "type": "Organization", "aliases": ["Harbour Blend"]}]},
            {"entities": [{"name": "Harbour Blend", "type": "Product", "aliases": []}]}]
    canon, _, ambiguous = load_graph.resolve(docs, [])
    assert ambiguous == 1
    assert canon["thing:fernwick"]["id"] != canon["thing:harbourblend"]["id"]


def test_lucene_query_escapes_and_drops_stopwords():
    assert graph_search.lucene("What did Alta-Verde's (2025) price do?") == r"Alta\-Verde OR 2025 OR price"
    assert graph_search.lucene("is it") == "*"


def test_rel_hints_map_question_words_to_relationship_types():
    assert graph_search.rel_hints("Who introduced the client?") == ["INTRODUCED", "CLIENT_OF"]
    assert graph_search.rel_hints("When we met") == ["MET_WITH"]
    assert graph_search.rel_hints("metal prices") == []     # short hints match whole words only


def test_trim_drops_lines_far_below_the_best_match():
    qvec = config.embed("harbour blend wholesale price")
    lines = ["Harbour Blend wholesale price $28", "Harbour Blend price rose", "Ruth hired a barista in Millbrook"]
    kept, dropped = graph_search.trim(qvec, lines)
    assert kept[0] == lines[0] and lines[2] not in kept and dropped == 1


def test_claim_line_marks_superseded_and_modality():
    c = {"current": False, "newer_value": "$28.00 per kg", "newer_date": "2025-09-01", "older_value": None,
         "conflicting_value": None, "modality": "plan", "evidence": None, "date": "2025-01-06",
         "subject": "Harbour Blend", "attribute": "wholesale price", "value": "$26.50 per kg", "quote": "q",
         "path": "Operations/2025-01 Wholesale price list.md"}
    line = graph_search.claim_line(c)
    assert "(plan)" in line and "[superseded 2025-09-01 by: $28.00 per kg]" in line
    assert line.endswith("(Operations/2025-01 Wholesale price list.md)")


def test_fit_stops_at_budget():
    lines, used = graph_search.fit(["a" * 10, "b" * 10, "c" * 10], 22)
    assert lines == ["a" * 10, "b" * 10] and used == 22
