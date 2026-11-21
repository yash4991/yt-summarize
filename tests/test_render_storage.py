import json

from interview_prep import render
from interview_prep.exporters import markdown_to_blocks
from interview_prep.storage import atomic_write_text, safe_name, video_folder


def test_safe_name_windows_rules():
    assert safe_name('What is "CAP"? A/B: test|x*') == "What is CAP A B test x"
    assert safe_name("CON") == "_CON"
    assert safe_name("   ...   ") == "untitled"
    long = safe_name("word " * 40, max_len=30)
    assert len(long) <= 30 and not long.endswith(" ")


def test_video_folder_avoids_collisions(tmp_path):
    a = video_folder(tmp_path, "Same Title", "aaaaaaaaaaa")
    (a / "metadata.json").write_text(json.dumps({"video": {"video_id": "aaaaaaaaaaa"}}))
    assert video_folder(tmp_path, "Same Title", "aaaaaaaaaaa") == a  # same video reuses
    b = video_folder(tmp_path, "Same Title", "bbbbbbbbbbb")
    assert b != a and "bbbbbbbbbbb" in b.name


def test_atomic_write(tmp_path):
    p = tmp_path / "x" / "f.txt"
    atomic_write_text(p, "hello")
    assert p.read_text() == "hello"
    assert not list(p.parent.glob(".tmp-*"))


def test_short_txt_has_bullets_and_valid_timestamps(pack, meta):
    out = render.render_short(pack, meta, "txt")
    assert "KEY POINTS" in out
    assert "• Cache hits avoid the database.  [01:05]" in out
    assert "99999" not in out and "Bad timestamp is dropped." in out  # invalid ts removed, point kept
    assert meta.watch_url in out


def test_short_md_links_timestamps(pack, meta):
    out = render.render_short(pack, meta, "md")
    assert "([01:05](https://youtu.be/abcdefghijk?t=65))" in out
    assert out.startswith("# Caching")


def test_detailed_paragraphs_and_sections(pack, meta):
    out = render.render_detailed(pack, meta, "txt")
    assert "Para one.\n\nPara two." in out
    assert "1. EVICTION" in out and "BEYOND THE VIDEO" in out


def test_qa_sorted_easy_first(pack, meta):
    out = render.render_qa(pack, meta, "md")
    assert out.index("What is a cache?") < out.index("What is a hard cache problem?")
    assert "[05:00](https://youtu.be/abcdefghijk?t=300)" in out


def test_markdown_to_notion_blocks():
    blocks = markdown_to_blocks("# T\n\n## H\n\n- a **b** [c](http://x)\n1. n\n\ntext\n\n---")
    types = [b["type"] for b in blocks]
    assert types == ["heading_1", "heading_2", "bulleted_list_item", "numbered_list_item", "paragraph", "divider"]
    rt = blocks[2]["bulleted_list_item"]["rich_text"]
    assert rt[1]["annotations"]["bold"] is True
    assert rt[3]["text"]["link"]["url"] == "http://x"
