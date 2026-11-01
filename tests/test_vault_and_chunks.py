"""The sample vault parses, and chunking follows the documented rule."""

import json
from pathlib import Path

import build_chunks
import config
import import_vault

ROOT = Path(__file__).resolve().parents[1]


def test_sample_vault_has_54_notes_and_skips_hidden_dirs(tmp_path):
    notes = list(import_vault.iter_markdown(config.vault_path()))
    assert len(notes) == 54
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "x.md").write_text("plugin config")
    (tmp_path / "a.md").write_text("note")
    (tmp_path / "b.txt").write_text("not markdown")
    assert [p.name for p in import_vault.iter_markdown(tmp_path)] == ["a.md"]


def test_parse_note_reads_frontmatter_links_and_tags(tmp_path):
    (tmp_path / "Clients").mkdir()
    md = tmp_path / "Clients" / "Cafe.md"
    md.write_text("---\ntags: [client, wholesale]\naliases: [Cafe X]\ndate: 2025-03-01\n---\n"
                  "# Cafe\nSee [[Harbour Blend|the blend]] and [[2025-01 Price list#Espresso]].\n"
                  "Photo ![[shop.png]]. #Active and `#not-a-tag` in code.\n")
    n = import_vault.parse_note(tmp_path, md)
    assert n["path"] == "Clients/Cafe.md" and n["folder"] == "Clients" and n["title"] == "Cafe"
    assert n["links"] == ["Harbour Blend", "2025-01 Price list"]       # alias and heading stripped, image ignored
    assert n["tags"] == ["active", "client", "wholesale"]
    assert n["aliases"] == ["Cafe X"]
    assert n["props"]["fm_date"] == "2025-03-01"


def test_parse_frontmatter_tolerates_bad_yaml():
    fm, body = import_vault.parse_frontmatter("---\n: [unclosed\n---\nbody")
    assert fm == {} and body == "body"
    assert import_vault.parse_frontmatter("no front matter") == ({}, "no front matter")


def test_short_note_is_one_chunk():
    assert build_chunks.chunk_note("T", "Just a line.") == [([], "Just a line.")]
    long_but_whole = "# A\n\n" + "word " * 900 + "\n\n# B\n\n" + "word " * 200     # ~1,400 tokens
    assert build_chunks.chunk_note("T", long_but_whole) == [([], long_but_whole.strip())]
    assert build_chunks.chunk_note("T", "   \n") == []


def test_long_note_splits_at_headings_within_limits():
    para = ("word " * 300).strip()                                       # ~375 tokens
    body = "\n\n".join([
        "# Plan", para, "## Pricing", para, para, "## Suppliers", para, para,
        "# Results", para, "## Q1", para, "```\n# not a heading\n```", "## Q2", "tiny",
    ])
    assert build_chunks.est_tokens(body) > build_chunks.CHUNK_WHOLE_NOTE_MAX
    chunks = build_chunks.chunk_note("T", body)
    assert all(build_chunks.est_tokens(t) <= build_chunks.CHUNK_MAX for _, t in chunks)
    # Full sections that would overflow stay apart; the Results sections fit together and share their top heading
    # (the '#' line inside the code fence is not a heading, and the tiny Q2 joins its neighbour).
    assert [p for p, _ in chunks] == [["Plan"], ["Plan", "Pricing"], ["Plan", "Suppliers"], ["Results"]]
    assert chunks[-1][1].startswith("# Results") and chunks[-1][1].rstrip().endswith("tiny")
    assert "".join(t for _, t in chunks).count("word") == 7 * 300          # nothing lost


def test_oversized_section_splits_with_overlap():
    para = ("alpha beta gamma delta " * 200)[:3000]                       # 750 tokens
    pieces = build_chunks.split_long("\n\n".join([para, para, para]))
    assert len(pieces) >= 2
    assert all(build_chunks.est_tokens(p) <= build_chunks.CHUNK_MAX + build_chunks.CHUNK_OVERLAP_CHARS for p in pieces)


def test_est_tokens_counts_cjk_per_character():
    assert build_chunks.est_tokens("abcd" * 10) == 10
    assert build_chunks.est_tokens("咖啡豆") == 3


def test_note_date_prefers_frontmatter_then_title():
    assert build_chunks.note_date("Journal/2024-02-05.md", {}, 0) == "2024-02-05"
    assert build_chunks.note_date("x.md", {"date": "2023-07-01T10:00"}, 0) == "2023-07-01"
    assert build_chunks.note_date("Decisions/2025-08 Raise.md", {}, 0) == "2025-08"
    assert build_chunks.note_date("x.md", {}, 0) == "1970-01-01"


def test_sample_rows_match_hand_written_extraction(sample_rows):
    """Every chunk of the sample vault has an extraction keyed by its id, and vice versa: a chunking change that
    moves boundaries would silently drop the hand-written graph."""
    extraction = json.loads((ROOT / "data" / "sample_extractions.json").read_text())
    ids = {r["id"] for r in sample_rows}
    assert len(sample_rows) == 54
    assert ids == set(extraction)
    r = next(r for r in sample_rows if r["note"] == "Journal/2024-02-05.md")
    assert r["breadcrumb"] == "2024-02-05" and r["note_date"] == "2024-02-05" and r["whole_note"]


def test_text_hash_changes_with_content_only():
    a = build_chunks.text_hash("T > H", "body")
    assert a == build_chunks.text_hash("T > H", "body")
    assert a != build_chunks.text_hash("T > H", "body!") and a != build_chunks.text_hash("T", "body")
